# This file contains code derived from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

from __future__ import annotations

import logging

import torch

from flag_gems.fused.FLA.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
from flag_gems.fused.FLA.utils import SUPPRESS_LEVEL

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_CU_SEQLENS_DTYPES = (torch.int32, torch.int64)
_CHUNK_SIZE = 64


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must have ranks [B, T, H, K], [B, T, H, K], "
            "and [B, T, HV, V]."
        )
    if g.ndim != 3:
        raise ValueError("g must have rank [B, T, HV].")
    if beta.ndim not in (3, 4):
        raise ValueError("beta must have rank [B, T, HV] or [B, T, HV, V].")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}.")

    B, T, H, K = q.shape
    Bv, Tv, HV, V = v.shape
    if (Bv, Tv) != (B, T):
        raise ValueError("v must share the same [B, T] dimensions as q and k.")
    if g.shape != (B, T, HV):
        raise ValueError(f"g must have shape {(B, T, HV)}, got {tuple(g.shape)}.")
    if beta.ndim == 3 and beta.shape != (B, T, HV):
        raise ValueError(f"beta must have shape {(B, T, HV)}, got {tuple(beta.shape)}.")
    if beta.ndim == 4 and beta.shape != (B, T, HV, V):
        raise ValueError(
            f"headwise beta must have shape {(B, T, HV, V)}, got {tuple(beta.shape)}."
        )
    if H <= 0 or HV <= 0 or HV % H != 0:
        raise ValueError(f"HV must be a positive multiple of H, got H={H}, HV={HV}.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if K > 128:
        raise ValueError("chunk_gated_delta_rule_fwd currently supports K <= 128.")
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must share dtype float16, bfloat16, or float32.")
    if g.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("g must have dtype float16, bfloat16, or float32.")
    if beta.dtype != q.dtype:
        raise TypeError("beta must use the same dtype as q, k, and v.")
    devices = {q.device, k.device, v.device, g.device, beta.device}
    if len(devices) != 1:
        raise ValueError("q, k, v, g, and beta must be on the same device.")
    if q.device.type != "cuda":
        raise ValueError("chunk_gated_delta_rule_fwd currently requires a CUDA device.")

    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a rank-1 tensor.")
        if cu_seqlens.dtype not in _SUPPORTED_CU_SEQLENS_DTYPES:
            raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64.")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q.")
        if B != 1:
            raise ValueError("B must be 1 when cu_seqlens is provided.")
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least a start and an end offset.")
        N = cu_seqlens.numel() - 1
    else:
        N = B

    if initial_state is not None:
        expected = (N, HV, K, V)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial_state must have shape {expected}, got {tuple(initial_state.shape)}."
            )
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q.")
        if initial_state.dtype not in _SUPPORTED_DTYPES:
            raise TypeError("initial_state must have dtype float16, bfloat16, or float32.")

    return B, T, H, HV, K, V, N


def _cu_seqlens_to_list(
    T: int,
    B: int,
    cu_seqlens: torch.Tensor | None,
) -> list[int]:
    if cu_seqlens is None:
        return [i * T for i in range(B + 1)]
    # A single host sync is kept here because the Python wrapper constructs
    # per-sequence state indices and the torch A workspace by segment.
    cu_cpu = cu_seqlens.detach().cpu().tolist()
    if cu_cpu[0] != 0 or cu_cpu[-1] != T:
        raise ValueError("cu_seqlens must start at 0 and end at q.shape[1].")
    if any(e < s for s, e in zip(cu_cpu[:-1], cu_cpu[1:])):
        raise ValueError("cu_seqlens must be non-decreasing.")
    return [int(x) for x in cu_cpu]


def _build_cu_seqlens(
    B: int,
    T: int,
    cu_seqlens: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if cu_seqlens is not None:
        return cu_seqlens.contiguous()
    return torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.long)


def _build_state_indices(offsets: list[int], device: torch.device) -> torch.Tensor:
    lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:])]
    max_len = max(lengths)
    indices = torch.empty((len(lengths), max_len), device=device, dtype=torch.long)
    for i, length in enumerate(lengths):
        if length > 0:
            indices[i, :length] = i
        if length < max_len:
            indices[i, length:] = i
    return indices.contiguous()


def _flatten_equal_length(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    return x.contiguous().reshape(1, B * T, *x.shape[2:])


def _prepare_recurrent_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        return (
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
        )
    B, T = q.shape[:2]
    return (
        _flatten_equal_length(q, B, T),
        _flatten_equal_length(k, B, T),
        _flatten_equal_length(v, B, T),
        _flatten_equal_length(g, B, T),
        _flatten_equal_length(beta, B, T),
    )


def _chunk_local_cumsum(
    g: torch.Tensor,
    offsets: list[int],
    chunk_size: int = _CHUNK_SIZE,
) -> torch.Tensor:
    g_out = torch.empty_like(g)
    if g.shape[0] > 1:
        for start in range(0, g.shape[1], chunk_size):
            end = min(start + chunk_size, g.shape[1])
            g_out[:, start:end, :] = torch.cumsum(g[:, start:end, :], dim=1)
        return g_out

    for seq_start, seq_end in zip(offsets[:-1], offsets[1:]):
        for start in range(seq_start, seq_end, chunk_size):
            end = min(start + chunk_size, seq_end)
            g_out[:, start:end, :] = torch.cumsum(g[:, start:end, :], dim=1)
    return g_out


def _solve_lower_unit_inverse(base: torch.Tensor) -> torch.Tensor:
    attn = -torch.tril(base, diagonal=-1)
    size = attn.shape[-1]
    for i in range(1, size):
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row[..., :, None] * attn[..., :i, :i]).sum(dim=-2)
    attn = attn + torch.eye(size, device=base.device, dtype=torch.float32)
    return attn


def _repeat_k_heads(k: torch.Tensor, HV: int) -> torch.Tensor:
    return k.repeat_interleave(HV // k.shape[2], dim=2)


def _compute_scalar_beta_a_chunk(
    k_hv: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    k_hv = k_hv.float().permute(0, 2, 1, 3)
    beta = beta.float().permute(0, 2, 1)
    g_cumsum = g_cumsum.float().permute(0, 2, 1)
    k_beta = k_hv * beta[..., None]
    kkt = k_beta @ k_hv.transpose(-1, -2)
    decay = torch.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :])
    return _solve_lower_unit_inverse(kkt * decay)


def _compute_scalar_beta_a(
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    offsets: list[int],
    chunk_size: int = _CHUNK_SIZE,
) -> torch.Tensor | None:
    if beta.ndim == 4:
        return None

    B, T, _H, _K = k.shape
    HV = beta.shape[2]
    k_hv = _repeat_k_heads(k, HV)
    A = torch.zeros((B, T, HV, chunk_size), device=k.device, dtype=k.dtype)

    if B > 1:
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            length = chunk_end - chunk_start
            solved = _compute_scalar_beta_a_chunk(
                k_hv[:, chunk_start:chunk_end],
                g_cumsum[:, chunk_start:chunk_end],
                beta[:, chunk_start:chunk_end],
            )
            A[:, chunk_start:chunk_end, :, :length] = solved.permute(0, 2, 1, 3).to(
                k.dtype
            )
        return A

    for seq_start, seq_end in zip(offsets[:-1], offsets[1:]):
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            length = chunk_end - chunk_start
            solved = _compute_scalar_beta_a_chunk(
                k_hv[:, chunk_start:chunk_end],
                g_cumsum[:, chunk_start:chunk_end],
                beta[:, chunk_start:chunk_end],
            )
            A[:, chunk_start:chunk_end, :, :length] = solved.permute(0, 2, 1, 3).to(
                k.dtype
            )
    return A


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
):
    """Forward-only chunk gated delta rule.

    Returns the FLA-compatible tuple ``(g, o, A, final_state, ..., ..., ...)``.
    ``A`` is the solved intra-chunk workspace for scalar beta. For V-dependent
    beta, FLA's ``[B, T, HV, 64]`` workspace cannot represent the per-value
    recurrence, so ``A`` is returned as ``None`` instead of a misleading tensor.
    """
    logger.debug("GEMS CHUNK GATED DELTA RULE FWD")
    B, T, _H, HV, K, V, N = _validate_inputs(q, k, v, g, beta, initial_state, cu_seqlens)
    if SUPPRESS_LEVEL >= 3:
        raise RuntimeError(
            "SUPPRESS_LEVEL >= 3 requests w/h/v_new debug tensors, which are not "
            "materialized by the recurrent chunk_gated_delta_rule_fwd path."
        )
    offsets = _cu_seqlens_to_list(T, B, cu_seqlens)
    lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:])]
    if any(length == 0 for length in lengths):
        raise ValueError("zero-length sequences in cu_seqlens are not supported.")

    q_work, k_work, v_work, g_work, beta_work = _prepare_recurrent_inputs(
        q, k, v, g, beta, cu_seqlens
    )
    cu_work = _build_cu_seqlens(B, T, cu_seqlens, q.device)
    state_indices = _build_state_indices(offsets, q.device)

    if initial_state is None:
        recurrent_initial = torch.zeros(
            (N, HV, K, V), device=q.device, dtype=torch.float32
        )
    else:
        recurrent_initial = initial_state.detach().to(torch.float32).contiguous().clone()

    o_work, recurrent_final = fused_recurrent_gated_delta_rule_fwd(
        q=q_work,
        k=k_work,
        v=v_work,
        g=g_work,
        beta=beta_work,
        scale=float(scale),
        initial_state=recurrent_initial,
        inplace_final_state=True,
        cu_seqlens=cu_work,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )

    if cu_seqlens is None:
        o = o_work.reshape(B, T, HV, V)
    else:
        o = o_work

    g_out = _chunk_local_cumsum(g.contiguous(), offsets)
    A = _compute_scalar_beta_a(k.contiguous(), g_out, beta.contiguous(), offsets)
    final_state = recurrent_final if output_final_state else None

    return g_out, o, A, final_state, None, None, None

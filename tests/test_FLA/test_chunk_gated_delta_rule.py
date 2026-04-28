from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flag_gems


CUDA_AVAILABLE = torch.cuda.is_available() and flag_gems.device == "cuda"


@dataclass(frozen=True)
class ChunkCase:
    name: str
    B: int
    T: int
    H: int
    HV: int
    K: int
    V: int
    dtype: torch.dtype
    beta_has_dim_v: bool = False
    use_initial_state: bool = False
    cu_seqlens: tuple[int, ...] | None = None
    non_contiguous_qkv: bool = False


CASES = [
    ChunkCase("tiny_fp16", 1, 1, 1, 1, 16, 16, torch.float16),
    ChunkCase("tiny_fp32", 1, 4, 2, 2, 16, 16, torch.float32),
    ChunkCase(
        "tiny_noncontig_bf16",
        1,
        4,
        2,
        2,
        32,
        32,
        torch.bfloat16,
        use_initial_state=True,
        non_contiguous_qkv=True,
    ),
    ChunkCase(
        "medium_fp16",
        2,
        128,
        4,
        4,
        64,
        64,
        torch.float16,
        use_initial_state=True,
    ),
    ChunkCase(
        "fla_like_reduced_bf16",
        1,
        256,
        4,
        8,
        128,
        128,
        torch.bfloat16,
        use_initial_state=True,
    ),
    ChunkCase(
        "varlen_headwise_beta_fp16",
        1,
        9,
        2,
        4,
        32,
        32,
        torch.float16,
        beta_has_dim_v=True,
        use_initial_state=True,
        cu_seqlens=(0, 2, 5, 9),
    ),
]


def _randn(shape: tuple[int, ...], dtype: torch.dtype, non_contiguous: bool = False):
    if not non_contiguous:
        return torch.randn(shape, device=flag_gems.device, dtype=dtype)
    storage_shape = (*shape[:-1], shape[-1] * 2)
    return torch.randn(storage_shape, device=flag_gems.device, dtype=dtype)[..., ::2]


def _build_inputs(case: ChunkCase):
    torch.manual_seed(20260428 + len(case.name))
    q = _randn((case.B, case.T, case.H, case.K), case.dtype, case.non_contiguous_qkv)
    k = _randn((case.B, case.T, case.H, case.K), case.dtype, case.non_contiguous_qkv)
    v = _randn((case.B, case.T, case.HV, case.V), case.dtype, case.non_contiguous_qkv)
    q.mul_(case.K**-0.5)
    k.mul_(case.K**-0.5)
    v.mul_(0.5)
    g = F.logsigmoid(
        torch.randn((case.B, case.T, case.HV), device=flag_gems.device, dtype=case.dtype)
    )
    if case.beta_has_dim_v:
        beta = torch.rand(
            (case.B, case.T, case.HV, case.V),
            device=flag_gems.device,
            dtype=case.dtype,
        ).sigmoid()
    else:
        beta = torch.rand(
            (case.B, case.T, case.HV), device=flag_gems.device, dtype=case.dtype
        ).sigmoid()

    cu_seqlens = None
    nseq = case.B
    if case.cu_seqlens is not None:
        cu_seqlens = torch.tensor(
            case.cu_seqlens, device=flag_gems.device, dtype=torch.long
        )
        nseq = len(case.cu_seqlens) - 1

    initial_state = None
    if case.use_initial_state:
        initial_state = 0.1 * torch.randn(
            (nseq, case.HV, case.K, case.V),
            device=flag_gems.device,
            dtype=case.dtype,
        )
        initial_state.mul_(0.1)
    return q, k, v, g, beta, initial_state, cu_seqlens


def _repeat_qk_heads(x: torch.Tensor, HV: int) -> torch.Tensor:
    group = HV // x.shape[-2]
    return x.repeat_interleave(group, dim=-2)


def _reference_equal_length(q, k, v, g, beta, scale, initial_state):
    B, T, _, _ = q.shape
    HV = v.shape[2]
    q_hv = _repeat_qk_heads(q.float(), HV)
    k_hv = _repeat_qk_heads(k.float(), HV)
    state = (
        torch.zeros(
            (B, HV, k.shape[-1], v.shape[-1]), device=q.device, dtype=torch.float32
        )
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty((B, T, HV, v.shape[-1]), device=q.device, dtype=torch.float32)
    for t in range(T):
        state = state * g[:, t].float().exp()[..., None, None]
        delta = v[:, t].float() - torch.einsum("bhkv,bhk->bhv", state, k_hv[:, t])
        if beta.ndim == 3:
            delta = delta * beta[:, t].float()[..., None]
        else:
            delta = delta * beta[:, t].float()
        state = state + k_hv[:, t][..., None] * delta[..., None, :]
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_hv[:, t] * scale, state)
    return out.to(v.dtype), state


def _reference_varlen(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    offsets = cu_seqlens.detach().cpu().tolist()
    N = len(offsets) - 1
    HV = v.shape[2]
    state = (
        torch.zeros(
            (N, HV, k.shape[-1], v.shape[-1]), device=q.device, dtype=torch.float32
        )
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty_like(v, dtype=torch.float32)
    for n, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        q_hv = _repeat_qk_heads(q[:, start:end].float(), HV)[0]
        k_hv = _repeat_qk_heads(k[:, start:end].float(), HV)[0]
        for i, pos in enumerate(range(start, end)):
            state[n] = state[n] * g[0, pos].float().exp()[..., None, None]
            delta = v[0, pos].float() - torch.einsum("hkv,hk->hv", state[n], k_hv[i])
            if beta.ndim == 3:
                delta = delta * beta[0, pos].float()[..., None]
            else:
                delta = delta * beta[0, pos].float()
            state[n] = state[n] + k_hv[i][..., None] * delta[..., None, :]
            out[0, pos] = torch.einsum("hk,hkv->hv", q_hv[i] * scale, state[n])
    return out.to(v.dtype), state


def _reference_chunk_local_cumsum(g, cu_seqlens, chunk_size=64):
    out = torch.empty_like(g)
    if cu_seqlens is None:
        for start in range(0, g.shape[1], chunk_size):
            end = min(start + chunk_size, g.shape[1])
            out[:, start:end] = torch.cumsum(g[:, start:end], dim=1)
        return out
    offsets = cu_seqlens.detach().cpu().tolist()
    for seq_start, seq_end in zip(offsets[:-1], offsets[1:]):
        for start in range(seq_start, seq_end, chunk_size):
            end = min(start + chunk_size, seq_end)
            out[:, start:end] = torch.cumsum(g[:, start:end], dim=1)
    return out


def _reference(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    if cu_seqlens is None:
        return _reference_equal_length(q, k, v, g, beta, scale, initial_state)
    return _reference_varlen(q, k, v, g, beta, scale, initial_state, cu_seqlens)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="chunk_gated_delta_rule requires CUDA")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_chunk_gated_delta_rule_matches_torch_reference(case: ChunkCase):
    q, k, v, g, beta, initial_state, cu_seqlens = _build_inputs(case)
    scale = case.K**-0.5

    g_out, out, A, final_state, *_ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ref_out, ref_final = _reference(q, k, v, g, beta, scale, initial_state, cu_seqlens)
    ref_g = _reference_chunk_local_cumsum(g, cu_seqlens)

    assert A.shape == (case.B, case.T, case.HV, 64)
    assert A.dtype == case.dtype
    assert torch.isfinite(out).all()
    assert torch.isfinite(final_state).all()
    torch.testing.assert_close(g_out, ref_g, rtol=0, atol=0)
    torch.testing.assert_close(out, ref_out, rtol=5e-2, atol=8e-2)
    torch.testing.assert_close(final_state, ref_final, rtol=5e-2, atol=8e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="chunk_gated_delta_rule requires CUDA")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_omits_final_state_when_not_requested():
    case = ChunkCase("no_final", 1, 4, 2, 2, 16, 16, torch.float16)
    q, k, v, g, beta, initial_state, cu_seqlens = _build_inputs(case)
    result = flag_gems.chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        case.K**-0.5,
        initial_state,
        False,
        cu_seqlens,
    )
    assert result[1].shape == (case.B, case.T, case.HV, case.V)
    assert result[3] is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="chunk_gated_delta_rule requires CUDA")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_rejects_invalid_shapes():
    q = torch.randn((1, 2, 2, 16), device=flag_gems.device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn((1, 2, 3, 16), device=flag_gems.device, dtype=torch.float16)
    g = torch.randn((1, 2, 3), device=flag_gems.device, dtype=torch.float16)
    beta = torch.randn((1, 2, 3), device=flag_gems.device, dtype=torch.float16)

    with pytest.raises(ValueError, match="HV"):
        flag_gems.chunk_gated_delta_rule_fwd(q, k, v, g, beta, 1.0, None, False)

    q_big = torch.randn((1, 1, 1, 129), device=flag_gems.device, dtype=torch.float16)
    with pytest.raises(ValueError, match="K <= 128"):
        flag_gems.chunk_gated_delta_rule_fwd(
            q_big,
            q_big,
            torch.randn((1, 1, 1, 16), device=flag_gems.device, dtype=torch.float16),
            torch.randn((1, 1, 1), device=flag_gems.device, dtype=torch.float16),
            torch.randn((1, 1, 1), device=flag_gems.device, dtype=torch.float16),
            1.0,
            None,
            False,
        )

    bad_cu = torch.tensor([0, 1], device=flag_gems.device, dtype=torch.long)
    with pytest.raises(ValueError, match="end at q.shape"):
        flag_gems.chunk_gated_delta_rule_fwd(q, k, v[:, :, :2], g[:, :, :2], beta[:, :, :2], 1.0, None, False, bad_cu)

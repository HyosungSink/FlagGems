from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flag_gems


CUDA_AVAILABLE = torch.cuda.is_available() and flag_gems.device == "cuda"
CHUNK_SIZE = 64


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
    g_dtype: torch.dtype | None = None
    beta_has_dim_v: bool = False
    use_initial_state: bool = False
    cu_seqlens: tuple[int, ...] | None = None
    cu_dtype: torch.dtype = torch.int64
    non_contiguous_qkv: bool = False


CASES = [
    ChunkCase("tiny_fp16", 1, 1, 1, 1, 16, 16, torch.float16),
    ChunkCase("tiny_fp32_strict", 1, 4, 2, 2, 16, 16, torch.float32),
    ChunkCase(
        "mixed_g_fp32_fp16_T65",
        1,
        65,
        2,
        4,
        32,
        32,
        torch.float16,
        g_dtype=torch.float32,
        use_initial_state=True,
    ),
    ChunkCase(
        "mixed_g_fp32_bf16_T129",
        1,
        129,
        4,
        4,
        64,
        64,
        torch.bfloat16,
        g_dtype=torch.float32,
        use_initial_state=True,
    ),
    ChunkCase(
        "medium_B2_fp16",
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
        "fla_like_K128_bf16",
        1,
        128,
        4,
        8,
        128,
        128,
        torch.bfloat16,
        use_initial_state=True,
    ),
    ChunkCase(
        "varlen_int32_cross64_fp16",
        1,
        129,
        2,
        4,
        32,
        32,
        torch.float16,
        use_initial_state=True,
        cu_seqlens=(0, 65, 129),
        cu_dtype=torch.int32,
    ),
    ChunkCase(
        "varlen_int64_headwise_beta_fp16",
        1,
        129,
        2,
        4,
        32,
        32,
        torch.float16,
        beta_has_dim_v=True,
        use_initial_state=True,
        cu_seqlens=(0, 63, 129),
        cu_dtype=torch.int64,
    ),
    ChunkCase(
        "noncontig_bf16",
        1,
        8,
        2,
        2,
        32,
        32,
        torch.bfloat16,
        use_initial_state=True,
        non_contiguous_qkv=True,
    ),
]


def _randn(shape: tuple[int, ...], dtype: torch.dtype, non_contiguous: bool = False):
    if not non_contiguous:
        return torch.randn(shape, device=flag_gems.device, dtype=dtype)
    storage_shape = (*shape[:-1], shape[-1] * 2)
    return torch.randn(storage_shape, device=flag_gems.device, dtype=dtype)[..., ::2]


def _build_inputs(case: ChunkCase):
    torch.manual_seed(20260428 + len(case.name))
    g_dtype = case.g_dtype or case.dtype
    q = _randn((case.B, case.T, case.H, case.K), case.dtype, case.non_contiguous_qkv)
    k = _randn((case.B, case.T, case.H, case.K), case.dtype, case.non_contiguous_qkv)
    v = _randn((case.B, case.T, case.HV, case.V), case.dtype, case.non_contiguous_qkv)
    q.mul_(case.K**-0.5)
    k.mul_(case.K**-0.5)
    v.mul_(0.5)
    g = F.logsigmoid(
        torch.randn((case.B, case.T, case.HV), device=flag_gems.device, dtype=g_dtype)
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
            case.cu_seqlens, device=flag_gems.device, dtype=case.cu_dtype
        )
        nseq = len(case.cu_seqlens) - 1

    initial_state = None
    if case.use_initial_state:
        initial_state = 0.01 * torch.randn(
            (nseq, case.HV, case.K, case.V),
            device=flag_gems.device,
            dtype=case.dtype,
        )
    return q, k, v, g, beta, initial_state, cu_seqlens


def _repeat_qk_heads(x: torch.Tensor, HV: int) -> torch.Tensor:
    return x.repeat_interleave(HV // x.shape[-2], dim=-2)


def _segments(B: int, T: int, cu_seqlens: torch.Tensor | None):
    if cu_seqlens is None:
        return [(batch, batch, 0, T) for batch in range(B)]
    offsets = cu_seqlens.detach().cpu().tolist()
    return [(0, seq, int(start), int(end)) for seq, (start, end) in enumerate(zip(offsets[:-1], offsets[1:]))]


def _reference_chunk_local_cumsum(g, cu_seqlens, chunk_size=CHUNK_SIZE):
    out = torch.empty_like(g)
    if cu_seqlens is None:
        for start in range(0, g.shape[1], chunk_size):
            end = min(start + chunk_size, g.shape[1])
            out[:, start:end] = torch.cumsum(g[:, start:end], dim=1)
        return out
    offsets = cu_seqlens.detach().cpu().tolist()
    for seq_start, seq_end in zip(offsets[:-1], offsets[1:]):
        for start in range(int(seq_start), int(seq_end), chunk_size):
            end = min(start + chunk_size, int(seq_end))
            out[:, start:end] = torch.cumsum(g[:, start:end], dim=1)
    return out


def _solve_lower_unit_inverse_reference(base: torch.Tensor) -> torch.Tensor:
    attn = -torch.tril(base, diagonal=-1)
    size = attn.shape[-1]
    for i in range(1, size):
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row[..., :, None] * attn[..., :i, :i]).sum(dim=-2)
    return attn + torch.eye(size, device=base.device, dtype=torch.float32)


def _reference_scalar_a_chunk(k_hv, g_cumsum, beta):
    k_hv = k_hv.float().permute(0, 2, 1, 3)
    beta = beta.float().permute(0, 2, 1)
    g_cumsum = g_cumsum.float().permute(0, 2, 1)
    k_beta = k_hv * beta[..., None]
    kkt = k_beta @ k_hv.transpose(-1, -2)
    decay = torch.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :])
    return _solve_lower_unit_inverse_reference(kkt * decay)


def _reference_scalar_chunk(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    B, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    g_cumsum = _reference_chunk_local_cumsum(g, cu_seqlens)
    q_hv = _repeat_qk_heads(q.float(), HV)
    k_hv = _repeat_qk_heads(k.float(), HV)
    nseq = B if cu_seqlens is None else cu_seqlens.numel() - 1
    state = (
        torch.zeros((nseq, HV, K, V), device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
    A = torch.zeros((B, T, HV, CHUNK_SIZE), device=q.device, dtype=k.dtype)

    for batch, seq, seq_start, seq_end in _segments(B, T, cu_seqlens):
        for chunk_start in range(seq_start, seq_end, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, seq_end)
            length = chunk_end - chunk_start
            q_chunk = q_hv[batch : batch + 1, chunk_start:chunk_end]
            k_chunk = k_hv[batch : batch + 1, chunk_start:chunk_end]
            v_chunk = v[batch : batch + 1, chunk_start:chunk_end].float()
            beta_chunk = beta[batch : batch + 1, chunk_start:chunk_end]
            g_chunk = g_cumsum[batch : batch + 1, chunk_start:chunk_end]

            solved = _reference_scalar_a_chunk(k_chunk, g_chunk, beta_chunk)
            A[batch : batch + 1, chunk_start:chunk_end, :, :length] = (
                solved.permute(0, 2, 1, 3).to(k.dtype)
            )

            q_by = q_chunk.permute(0, 2, 1, 3) * scale
            k_by = k_chunk.permute(0, 2, 1, 3)
            v_by = v_chunk.permute(0, 2, 1, 3)
            beta_by = beta_chunk.float().permute(0, 2, 1)
            g_by = g_chunk.float().permute(0, 2, 1)
            decay = torch.exp(g_by[..., :, None] - g_by[..., None, :]).tril()
            v_beta = v_by * beta_by[..., None]
            value = solved @ v_beta
            k_cumdecay = solved @ (k_by * beta_by[..., None] * g_by.exp()[..., None])
            state_seq = state[seq : seq + 1]
            v_prime = k_cumdecay @ state_seq
            v_new = value - v_prime
            attn = (q_by @ k_by.transpose(-1, -2)) * decay
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            attn = attn.masked_fill(mask, 0)
            attn_inter = (q_by * g_by.exp()[..., None]) @ state_seq
            out_chunk = attn_inter + attn @ v_new
            out[batch, chunk_start:chunk_end] = out_chunk[0].permute(1, 0, 2)
            state[seq : seq + 1] = (
                state_seq * g_by[..., -1, None, None].exp()
                + (
                    k_by
                    * torch.exp(g_by[..., -1, None] - g_by)[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

    return g_cumsum, out.to(v.dtype), A, state


def _reference_recurrent(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    B, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    g_cumsum = _reference_chunk_local_cumsum(g, cu_seqlens)
    q_hv = _repeat_qk_heads(q.float(), HV)
    k_hv = _repeat_qk_heads(k.float(), HV)
    nseq = B if cu_seqlens is None else cu_seqlens.numel() - 1
    state = (
        torch.zeros((nseq, HV, K, V), device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)

    for batch, seq, seq_start, seq_end in _segments(B, T, cu_seqlens):
        for pos in range(seq_start, seq_end):
            state[seq] = state[seq] * g[batch, pos].float().exp()[..., None, None]
            delta = v[batch, pos].float() - torch.einsum(
                "hkv,hk->hv", state[seq], k_hv[batch, pos]
            )
            if beta.ndim == 3:
                delta = delta * beta[batch, pos].float()[..., None]
            else:
                delta = delta * beta[batch, pos].float()
            state[seq] = state[seq] + k_hv[batch, pos][..., None] * delta[..., None, :]
            out[batch, pos] = torch.einsum(
                "hk,hkv->hv", q_hv[batch, pos] * scale, state[seq]
            )
    return g_cumsum, out.to(v.dtype), None, state


def _reference(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    if beta.ndim == 3:
        return _reference_scalar_chunk(q, k, v, g, beta, scale, initial_state, cu_seqlens)
    return _reference_recurrent(q, k, v, g, beta, scale, initial_state, cu_seqlens)


def _tol(dtype: torch.dtype, final_state: bool = False):
    if dtype == torch.float32:
        return {"rtol": 8e-4, "atol": 8e-4}
    if dtype == torch.float16:
        return {"rtol": 2e-2 if not final_state else 3e-2, "atol": 3e-2 if not final_state else 5e-2}
    return {"rtol": 3e-2 if not final_state else 4e-2, "atol": 5e-2 if not final_state else 7e-2}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="chunk_gated_delta_rule requires CUDA")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_chunk_gated_delta_rule_matches_fla_megatron_reference(case: ChunkCase):
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

    ref_g, ref_out, ref_A, ref_final = _reference(
        q, k, v, g, beta, scale, initial_state, cu_seqlens
    )

    assert out.dtype == case.dtype
    assert g_out.dtype == (case.g_dtype or case.dtype)
    assert final_state.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(final_state).all()
    torch.testing.assert_close(g_out, ref_g, rtol=0, atol=0)
    torch.testing.assert_close(out, ref_out, **_tol(case.dtype))
    torch.testing.assert_close(final_state, ref_final, **_tol(case.dtype, final_state=True))

    if beta.ndim == 3:
        assert A is not None
        assert A.shape == (case.B, case.T, case.HV, CHUNK_SIZE)
        assert A.dtype == case.dtype
        torch.testing.assert_close(A, ref_A, **_tol(case.dtype))
    else:
        assert A is None
        assert ref_A is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="chunk_gated_delta_rule requires CUDA")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_omits_final_state_when_not_requested():
    case = ChunkCase("no_final", 1, 65, 2, 4, 32, 32, torch.float16)
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
    ref_g, ref_out, ref_A, _ = _reference(
        q, k, v, g, beta, case.K**-0.5, initial_state, cu_seqlens
    )
    assert result[1].shape == (case.B, case.T, case.HV, case.V)
    assert result[3] is None
    torch.testing.assert_close(result[0], ref_g, rtol=0, atol=0)
    torch.testing.assert_close(result[1], ref_out, **_tol(case.dtype))
    torch.testing.assert_close(result[2], ref_A, **_tol(case.dtype))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="chunk_gated_delta_rule requires CUDA")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_rejects_invalid_shapes_and_cu_seqlens():
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

    bad_end = torch.tensor([0, 1], device=flag_gems.device, dtype=torch.int32)
    with pytest.raises(ValueError, match="end at q.shape"):
        flag_gems.chunk_gated_delta_rule_fwd(
            q, k, v[:, :, :2], g[:, :, :2], beta[:, :, :2], 1.0, None, False, bad_end
        )

    zero_len = torch.tensor([0, 0, 2], device=flag_gems.device, dtype=torch.int64)
    with pytest.raises(ValueError, match="zero-length"):
        flag_gems.chunk_gated_delta_rule_fwd(
            q, k, v[:, :, :2], g[:, :, :2], beta[:, :, :2], 1.0, None, False, zero_len
        )

    bad_dtype = torch.tensor([0, 2], device=flag_gems.device, dtype=torch.int16)
    with pytest.raises(TypeError, match="torch.int32 or torch.int64"):
        flag_gems.chunk_gated_delta_rule_fwd(
            q, k, v[:, :, :2], g[:, :, :2], beta[:, :, :2], 1.0, None, False, bad_dtype
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="chunk_gated_delta_rule requires CUDA")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_rejects_suppress_level_debug_path(monkeypatch):
    import flag_gems.fused.FLA.chunk as chunk_module

    case = ChunkCase("suppress", 1, 2, 1, 1, 16, 16, torch.float16)
    q, k, v, g, beta, initial_state, cu_seqlens = _build_inputs(case)
    monkeypatch.setattr(chunk_module, "SUPPRESS_LEVEL", 3)
    with pytest.raises(RuntimeError, match="SUPPRESS_LEVEL >= 3"):
        flag_gems.chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, case.K**-0.5, initial_state, True, cu_seqlens
        )

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flag_gems


CUDA_AVAILABLE = torch.cuda.is_available() and flag_gems.device == "cuda"
CHUNK_SIZE = 64


@dataclass(frozen=True)
class BenchShape:
    name: str
    B: int
    T: int
    H: int
    HV: int
    K: int
    V: int
    dtype: torch.dtype


SHAPES = [
    BenchShape("tiny", 1, 1, 1, 1, 16, 16, torch.float16),
    BenchShape("small", 1, 32, 4, 4, 64, 64, torch.float16),
    BenchShape("cross_chunk", 1, 65, 4, 4, 64, 64, torch.float16),
    BenchShape("medium", 1, 128, 4, 4, 64, 64, torch.float16),
    BenchShape("long", 1, 512, 4, 4, 64, 64, torch.bfloat16),
    BenchShape("fla_reduced", 1, 128, 4, 8, 128, 128, torch.bfloat16),
]


def _build_inputs(shape: BenchShape):
    torch.manual_seed(20260428 + shape.T + shape.K)
    device = flag_gems.device
    q = torch.randn((shape.B, shape.T, shape.H, shape.K), device=device, dtype=shape.dtype)
    k = torch.randn((shape.B, shape.T, shape.H, shape.K), device=device, dtype=shape.dtype)
    v = torch.randn((shape.B, shape.T, shape.HV, shape.V), device=device, dtype=shape.dtype)
    q.mul_(shape.K**-0.5)
    k.mul_(shape.K**-0.5)
    v.mul_(0.5)
    g = F.logsigmoid(
        torch.randn((shape.B, shape.T, shape.HV), device=device, dtype=shape.dtype)
    )
    beta = torch.rand(
        (shape.B, shape.T, shape.HV), device=device, dtype=shape.dtype
    ).sigmoid()
    initial_state = torch.zeros(
        (shape.B, shape.HV, shape.K, shape.V), device=device, dtype=shape.dtype
    )
    return q, k, v, g, beta, initial_state


def _repeat_qk_heads(x: torch.Tensor, HV: int) -> torch.Tensor:
    return x.float().repeat_interleave(HV // x.shape[-2], dim=-2)


def _chunk_local_cumsum(g: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    out = torch.empty_like(g)
    for start in range(0, g.shape[1], chunk_size):
        end = min(start + chunk_size, g.shape[1])
        out[:, start:end] = torch.cumsum(g[:, start:end], dim=1)
    return out


def _solve_lower_unit_inverse(base: torch.Tensor) -> torch.Tensor:
    attn = -torch.tril(base, diagonal=-1)
    size = attn.shape[-1]
    for i in range(1, size):
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row[..., :, None] * attn[..., :i, :i]).sum(dim=-2)
    return attn + torch.eye(size, device=base.device, dtype=torch.float32)


def _a_chunk(k_hv: torch.Tensor, g_cumsum: torch.Tensor, beta: torch.Tensor):
    k_hv = k_hv.float().permute(0, 2, 1, 3)
    beta = beta.float().permute(0, 2, 1)
    g_cumsum = g_cumsum.float().permute(0, 2, 1)
    k_beta = k_hv * beta[..., None]
    kkt = k_beta @ k_hv.transpose(-1, -2)
    decay = torch.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :])
    return _solve_lower_unit_inverse(kkt * decay)


def _megatron_torch_chunk_reference(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state=True,
):
    B, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    g_cumsum = _chunk_local_cumsum(g)
    q_hv = _repeat_qk_heads(q, HV)
    k_hv = _repeat_qk_heads(k, HV)
    state = initial_state.float().clone()
    out = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
    A = torch.zeros((B, T, HV, CHUNK_SIZE), device=q.device, dtype=k.dtype)

    for chunk_start in range(0, T, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, T)
        length = chunk_end - chunk_start
        q_chunk = q_hv[:, chunk_start:chunk_end]
        k_chunk = k_hv[:, chunk_start:chunk_end]
        v_chunk = v[:, chunk_start:chunk_end].float()
        beta_chunk = beta[:, chunk_start:chunk_end]
        g_chunk = g_cumsum[:, chunk_start:chunk_end]

        solved = _a_chunk(k_chunk, g_chunk, beta_chunk)
        A[:, chunk_start:chunk_end, :, :length] = solved.permute(0, 2, 1, 3).to(
            k.dtype
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
        v_prime = k_cumdecay @ state
        v_new = value - v_prime
        attn = (q_by @ k_by.transpose(-1, -2)) * decay
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        attn = attn.masked_fill(mask, 0)
        attn_inter = (q_by * g_by.exp()[..., None]) @ state
        out[:, chunk_start:chunk_end] = (attn_inter + attn @ v_new).permute(0, 2, 1, 3)
        state = (
            state * g_by[..., -1, None, None].exp()
            + (k_by * torch.exp(g_by[..., -1, None] - g_by)[..., None]).transpose(-1, -2)
            @ v_new
        )

    final = state if output_final_state else None
    return g_cumsum, out.to(v.dtype), A, final, None, None, None


def _gems_op(q, k, v, g, beta, scale, initial_state, output_final_state=True):
    return flag_gems.chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state, output_final_state
    )


def _time_ms(fn, args, iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return {"rtol": 8e-4, "atol": 8e-4}
    if dtype == torch.float16:
        return {"rtol": 2e-2, "atol": 3e-2}
    return {"rtol": 3e-2, "atol": 5e-2}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="benchmark requires CUDA")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("shape", SHAPES, ids=[shape.name for shape in SHAPES])
def test_perf_chunk_gated_delta_rule(shape: BenchShape):
    q, k, v, g, beta, initial_state = _build_inputs(shape)
    args = (q, k, v, g, beta, shape.K**-0.5, initial_state, True)

    with torch.no_grad():
        gems_result = _gems_op(*args)
        ref_result = _megatron_torch_chunk_reference(*args)
        torch.testing.assert_close(gems_result[0], ref_result[0], rtol=0, atol=0)
        torch.testing.assert_close(gems_result[1], ref_result[1], **_tol(shape.dtype))
        torch.testing.assert_close(gems_result[2], ref_result[2], **_tol(shape.dtype))
        torch.testing.assert_close(gems_result[3], ref_result[3], **_tol(shape.dtype))

        for _ in range(3):
            _gems_op(*args)
            _megatron_torch_chunk_reference(*args)
        gems_iters = 20 if shape.T <= 128 else 10
        ref_iters = 10 if shape.T <= 128 else 3
        gems_ms = _time_ms(_gems_op, args, gems_iters)
        ref_ms = _time_ms(_megatron_torch_chunk_reference, args, ref_iters)

    speedup = ref_ms / gems_ms if gems_ms > 0 else float("inf")
    print(
        "chunk_gated_delta_rule "
        f"shape={shape.name} B={shape.B} T={shape.T} H={shape.H} HV={shape.HV} "
        f"K={shape.K} V={shape.V} dtype={shape.dtype} "
        f"gems_ms={gems_ms:.3f} megatron_torch_chunk_ref_ms={ref_ms:.3f} "
        f"chunk_ref_speedup={speedup:.2f}x"
    )

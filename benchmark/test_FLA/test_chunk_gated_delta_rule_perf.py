from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flag_gems


CUDA_AVAILABLE = torch.cuda.is_available() and flag_gems.device == "cuda"


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


def _torch_reference(q, k, v, g, beta, scale, initial_state, output_final_state=True):
    B, T, _, _ = q.shape
    HV = v.shape[2]
    q_hv = _repeat_qk_heads(q, HV)
    k_hv = _repeat_qk_heads(k, HV)
    state = initial_state.float().clone()
    out = torch.empty((B, T, HV, v.shape[-1]), device=q.device, dtype=torch.float32)
    for t in range(T):
        state = state * g[:, t].float().exp()[..., None, None]
        delta = v[:, t].float() - torch.einsum("bhkv,bhk->bhv", state, k_hv[:, t])
        delta = delta * beta[:, t].float()[..., None]
        state = state + k_hv[:, t][..., None] * delta[..., None, :]
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_hv[:, t] * scale, state)
    final = state if output_final_state else None
    return None, out.to(v.dtype), None, final, None, None, None


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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="benchmark requires CUDA")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("shape", SHAPES, ids=[shape.name for shape in SHAPES])
def test_perf_chunk_gated_delta_rule(shape: BenchShape):
    q, k, v, g, beta, initial_state = _build_inputs(shape)
    args = (q, k, v, g, beta, shape.K**-0.5, initial_state, True)

    with torch.no_grad():
        gems_result = _gems_op(*args)
        ref_result = _torch_reference(*args)
        torch.testing.assert_close(gems_result[1], ref_result[1], rtol=5e-2, atol=8e-2)
        torch.testing.assert_close(gems_result[3], ref_result[3], rtol=5e-2, atol=8e-2)

        for _ in range(3):
            _gems_op(*args)
        gems_iters = 20 if shape.T <= 128 else 10
        ref_iters = 3 if shape.T <= 32 else 1
        gems_ms = _time_ms(_gems_op, args, gems_iters)
        ref_ms = _time_ms(_torch_reference, args, ref_iters)

    speedup = ref_ms / gems_ms if gems_ms > 0 else float("inf")
    print(
        "chunk_gated_delta_rule "
        f"shape={shape.name} B={shape.B} T={shape.T} H={shape.H} HV={shape.HV} "
        f"K={shape.K} V={shape.V} dtype={shape.dtype} "
        f"gems_ms={gems_ms:.3f} torch_ref_ms={ref_ms:.3f} speedup={speedup:.2f}x"
    )

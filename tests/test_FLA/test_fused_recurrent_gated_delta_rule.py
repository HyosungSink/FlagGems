import random
import os
from typing import Dict, List

import pytest
import torch
import torch.nn.functional as F

import flag_gems

# Importing vLLM after FlagGems can abort the CoreX process via ixformer duplicate
# native op registration. Keep the external reference opt-in; the torch fallback
# test below still exercises this file by default.
if os.getenv("FLAGGEMS_ENABLE_VLLM_REFERENCE", "0") == "1":
    try:
        from vllm.model_executor.layers.fla.ops import (
            fused_recurrent_gated_delta_rule as base_fused_recurrent_gated_delta_rule,
        )

        VLLM_AVAILABLE = True
    except Exception:  # pragma: no cover - optional dependency guard
        base_fused_recurrent_gated_delta_rule = None
        VLLM_AVAILABLE = False
else:
    base_fused_recurrent_gated_delta_rule = None
    VLLM_AVAILABLE = False

random.seed(42)
torch.manual_seed(42)


def is_cuda_available() -> bool:
    return torch.cuda.is_available() and flag_gems.device == "cuda"


CUDA_AVAILABLE = is_cuda_available()


def _repeat_qk_heads(x: torch.Tensor, HV: int) -> torch.Tensor:
    return x.float().repeat_interleave(HV // x.shape[-2], dim=-2)


def _torch_recurrent_reference(q, k, v, g, beta, scale, initial_state):
    B, T, _, _ = q.shape
    HV = v.shape[2]
    q_hv = _repeat_qk_heads(q, HV)
    k_hv = _repeat_qk_heads(k, HV)
    state = initial_state.float().clone()
    out = torch.empty_like(v, dtype=torch.float32)
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


def rearrange_mixed_qkv(
    mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim, tp_size=1, contiguous=True
):
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, query.shape[0], -1, head_k_dim)
    key = key.view(1, key.shape[0], -1, head_k_dim)
    value = value.view(1, value.shape[0], -1, head_v_dim)
    if contiguous:
        return query.contiguous(), key.contiguous(), value.contiguous()
    else:
        return query, key, value


class FusedRecurrentGatedDeltaRuleTestKit:
    base_dtype = torch.bfloat16

    @staticmethod
    def _cases() -> List[Dict]:
        cases = [
            {  # cu_seqlens situation
                "H": 16,  # global heads(aka key_dim); local = H / tp_size = 4
                "HV": 32,  # global value heads(aka value_dim); local = HV / tp_size = 8
                "K": 128,
                "V": 128,
                "tp_size": 4,
                "beta_has_dim_v": False,
                "inplace_final_state": True,
                "use_qk_l2norm": True,
                "scale": 0.08838834764831845,
                "ssm_state_len": 4589,
                "ssm_state_indices_all_zero": True,
                "cu_seqlens_explicit": True,
            },
        ]
        return cases

    @classmethod
    def get_test_params(cls) -> List[Dict]:
        return cls._cases()

    @classmethod
    def build_inputs(cls, cfg: Dict, T, qkv_contiguous: bool) -> Dict:
        device = flag_gems.device
        dtype = cls.base_dtype
        tp_size = cfg.get("tp_size", 1)

        B = 1  # for cu_seqlens inputs, batch size is 1 and cu_seqlens is required
        cu_seqlens_len = T + 1
        key_dim = cfg["H"] * cfg["K"]  # 16 * 128 = 2048
        value_dim = cfg["HV"] * cfg["V"]  # 32 * 128 = 4096

        assert key_dim % tp_size == 0, "key_dim must be divisible by tp_size"
        assert value_dim % tp_size == 0, "value_dim must be divisible by tp_size"
        assert (key_dim // tp_size) % cfg[
            "K"
        ] == 0, "(key_dim/tp_size) must be multiple of head_k_dim"
        assert (value_dim // tp_size) % cfg[
            "V"
        ] == 0, "(value_dim/tp_size) must be multiple of head_v_dim"

        # Build mixed_qkv with explicit (T, mixed_qkv_dim) shape. For the non-contiguous
        # branch we slice a strided view from a 3D buffer to simulate a real packing.
        mixed_qkv_dim = (2 * key_dim + value_dim) // tp_size
        total_tokens = B * T  # currently B=1, so this equals T
        mixed_qkv = torch.randn(
            (total_tokens, mixed_qkv_dim), device=device, dtype=dtype
        )

        query, key, value = rearrange_mixed_qkv(
            mixed_qkv,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=cfg["K"],
            head_v_dim=cfg["V"],
            tp_size=tp_size,
            contiguous=qkv_contiguous,
        )

        HV_local = value.shape[2]

        g = F.logsigmoid(torch.randn((B, T, HV_local), device=device, dtype=dtype))
        if cfg["beta_has_dim_v"]:
            beta = torch.rand(
                B, T, HV_local, cfg["V"], device=device, dtype=dtype
            ).sigmoid()
        else:
            beta = torch.rand(B, T, HV_local, device=device, dtype=dtype).sigmoid()

        cu_seqlens = torch.arange(cu_seqlens_len, device=device, dtype=torch.long)
        initial_state = torch.zeros(
            (cfg["ssm_state_len"], HV_local, cfg["K"], cfg["V"]),
            device=device,
            dtype=dtype,
        )
        if cfg.get("ssm_state_indices_all_zero", False):
            ssm_state_indices = torch.zeros(T, device=device, dtype=torch.long)
        else:
            ssm_state_indices = torch.arange(T, device=device, dtype=torch.long)

        scale = cfg["scale"] if cfg["scale"] is not None else cfg["K"] ** -0.5

        return {
            "q": query,
            "k": key,
            "v": value,
            "g": g,
            "beta": beta,
            "scale": float(scale),
            "initial_state": initial_state,
            "cu_seqlens": cu_seqlens,
            "inplace_final_state": cfg["inplace_final_state"],
            "use_qk_l2norm_in_kernel": cfg["use_qk_l2norm"],
            "ssm_state_indices": ssm_state_indices,
        }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("beta_has_dim_v", [False, True])
def test_fused_recurrent_gated_delta_rule_matches_torch_reference(beta_has_dim_v):
    torch.manual_seed(20260428 + int(beta_has_dim_v))
    B, T, H, HV, K, V = 1, 8, 2, 4, 16, 16
    dtype = torch.float16
    q = torch.randn((B, T, H, K), device=flag_gems.device, dtype=dtype) * (K**-0.5)
    k = torch.randn((B, T, H, K), device=flag_gems.device, dtype=dtype) * (K**-0.5)
    v = torch.randn((B, T, HV, V), device=flag_gems.device, dtype=dtype) * 0.5
    g = F.logsigmoid(torch.randn((B, T, HV), device=flag_gems.device, dtype=dtype))
    if beta_has_dim_v:
        beta = torch.rand((B, T, HV, V), device=flag_gems.device, dtype=dtype).sigmoid()
    else:
        beta = torch.rand((B, T, HV), device=flag_gems.device, dtype=dtype).sigmoid()
    initial_state = torch.zeros((B, HV, K, V), device=flag_gems.device, dtype=torch.float32)
    flag_initial = initial_state.clone()
    ref_initial = initial_state.clone()
    scale = K**-0.5
    cu_seqlens = torch.tensor([0, T], device=flag_gems.device, dtype=torch.long)
    ssm_state_indices = torch.zeros(T, device=flag_gems.device, dtype=torch.long)

    flag_out, flag_final = flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=flag_initial,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )
    ref_out, ref_final = _torch_recurrent_reference(
        q, k, v, g, beta, scale, ref_initial
    )

    torch.testing.assert_close(flag_out, ref_out, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(flag_final, ref_final, rtol=3e-2, atol=5e-2)


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and CUDA_AVAILABLE),
    reason="requires vLLM installed and CUDA device",
)
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("cfg", FusedRecurrentGatedDeltaRuleTestKit.get_test_params())
@pytest.mark.parametrize("T", [1, 2, 4, 128, 512])
@pytest.mark.parametrize("qkv_contiguous", [True, False])
def test_fused_recurrent_gated_delta_rule_matches_vllm(cfg, T, qkv_contiguous):
    kit = FusedRecurrentGatedDeltaRuleTestKit
    inputs = kit.build_inputs(cfg, T, qkv_contiguous)

    flag_initial = inputs["initial_state"].clone()
    base_initial = inputs["initial_state"].clone()

    flag_out, flag_final = flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=flag_initial,
        inplace_final_state=inputs["inplace_final_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    base_out, base_final = base_fused_recurrent_gated_delta_rule(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=base_initial,
        inplace_final_state=inputs["inplace_final_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    torch.testing.assert_close(flag_out, base_out, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(flag_final, base_final, rtol=1.5, atol=1.0)

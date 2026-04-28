# `chunk_gated_delta_rule` Implementation Plan

## Stage 0: Define Semantics

Before writing kernels, freeze the exact supported signature:

- Tensor inputs: `q`, `k`, `v`, `g`, `beta`.
- Optional inputs: `initial_state`, `cu_seqlens`, `output_final_state`, or equivalent.
- Shape convention: usually `[B, T, H, K]` for `q/k`, `[B, T, HV, V]` for `v`, and `[B, T, HV]` or `[B, T, HV, V]` for gates depending on reference.
- Supported dtype: start with `torch.float16` and `torch.bfloat16`; use `float32` reference accumulation when useful.
- Supported layout: contiguous first; add non-contiguous only after correctness is stable.

Create a local reference harness using FLA or Megatron Core and record the exact
reference function and version used.

## Stage 1: Correct Forward

Implement a minimal forward path that matches the reference for:

- Small sizes that are easy to debug.
- Representative FLA shapes.
- Multiple sequence lengths.
- Both `float16` and `bfloat16` if supported.
- Optional final state if the selected API exposes it.

Keep the first version simple enough to audit. A slower correct implementation
is better than a fast version whose recurrence is hard to verify.

## Stage 2: Integration

Choose one integration path and keep it consistent:

- `src/flag_gems/ops/chunk_gated_delta_rule.py` if matching current competition PR style.
- `src/flag_gems/fused/chunk_gated_delta_rule.py` if following existing FLA/fused organization.

Then update only the required exports/registrations:

- `src/flag_gems/ops/__init__.py` or `src/flag_gems/fused/__init__.py`
- `src/flag_gems/__init__.py` if the function is exposed through FlagGems

## Stage 3: Tests

Add tests before broad tuning:

- Deterministic random inputs with fixed seeds.
- Small exact-shape cases.
- Realistic FLA-like cases.
- Dtype coverage.
- Optional `initial_state` and final state checks.
- `cu_seqlens` or variable-length behavior if supported.

## Stage 4: Benchmark

Add benchmark coverage only after correctness is stable:

- Small sequence lengths to expose overhead.
- Medium and large sequence lengths to show throughput.
- Shapes taken from FLA/GatedDeltaNet usage when possible.
- Compare against FLA/Megatron reference or a torch-native fallback.

## Stage 5: PR Evidence

Before opening the PR, collect:

- Accuracy logs.
- Benchmark logs.
- Supported/unsupported case table.
- Exact reference version.
- A short explanation of why the implementation is maintainable.

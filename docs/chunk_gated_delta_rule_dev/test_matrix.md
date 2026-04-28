# `chunk_gated_delta_rule` Test Matrix

Use this as the coverage checklist for `tests/test_chunk_gated_delta_rule.py`
or `tests/test_FLA/test_chunk_gated_delta_rule.py`.

## Shape Coverage

- Tiny debug:
  - `B=1, T=1, H=1, HV=1, K=16, V=16`
  - `B=1, T=4, H=2, HV=2, K=32, V=32`
- Medium:
  - `B=1, T=64, H=4, HV=4, K=64, V=64`
  - `B=2, T=128, H=4, HV=4, K=64, V=64`
- FLA-like:
  - `B=1, T in [128, 256, 512]`
  - `H=16, HV=32, K=128, V=128` or the reduced local equivalent

## Dtype Coverage

- `torch.float16`
- `torch.bfloat16`
- `torch.float32` reference-only sanity check if useful
- `g=torch.float32` with `q/k/v` in `torch.float16` or `torch.bfloat16`

## Feature Coverage

- Contiguous `q/k/v`.
- Optional non-contiguous `q/k/v` views after the base path passes.
- `beta` shaped `[B, T, HV]`.
- `beta` shaped `[B, T, HV, V]` if supported.
- `initial_state is None`.
- `initial_state` provided.
- Final state returned or updated if supported.
- `cu_seqlens is None`.
- Explicit `cu_seqlens` for variable-length token streams if supported.
- `cu_seqlens` in both `torch.int32` and `torch.int64`.
- Cross-chunk sequence lengths such as `T=65` and `T=129`.
- Scalar beta returns a real `[B, T, HV, 64]` `A` workspace.
- Value-dependent beta returns `A is None` rather than a fake workspace.

## Numerical Checks

- Compare output with reference using dtype-aware tolerances.
- Compare final state separately because recurrence error can accumulate.
- Assert no NaN/Inf for normal random inputs.
- Use fixed seeds for reproducibility.

## Negative / Guard Checks

- Invalid ranks should raise clean errors.
- Incompatible head dimensions should raise clean errors.
- Unsupported options should fail explicitly rather than silently producing wrong results.

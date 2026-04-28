# [FlagGems Operator Development Competition] Add chunk_gated_delta_rule operator

## Summary

This PR adds forward support for `chunk_gated_delta_rule_fwd` in the existing
FlagGems FLA fused path. The implementation keeps the scope limited to this
operator, reuses the local recurrent gated delta rule Triton kernel for the
forward recurrence, and does not vendor external model repositories.

## Reference

Reference sources inspected:

- FLA pip packages: `flash-linear-attention==0.5.0`, `fla-core==0.5.0`.
- Megatron Core pip package: `megatron-core==0.16.1`.
- The current CoreX image has `torch==2.7.1+corex.4.4.0`.

In this environment, FLA package source is present, but importing
`fla.ops.gated_delta_rule.chunk` fails during Triton/autotuner setup with
`ValueError: 'BT' is not in list`. Megatron Core source is present, but the
runtime imports CoreX's preinstalled `megatron` package first, so
`megatron.core.ssm.gated_delta_net` is not importable by normal module lookup.
`inspect_reference.py` records both the import status and the source fallback.

## Supported Signature

```python
flag_gems.chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=None,
)
```

Supported inputs:

- `q`, `k`: `[B, T, H, K]`.
- `v`: `[B, T, HV, V]`.
- `g`: `[B, T, HV]`, log-space decay.
- `beta`: `[B, T, HV]` or `[B, T, HV, V]`.
- `initial_state`: `None` or `[N, HV, K, V]`.
- `cu_seqlens`: `None` or rank-1 `torch.long`; when provided, `B == 1`.
- Dtypes: `float16`, `bfloat16`, `float32`.
- Head grouping: `HV` must be a positive multiple of `H`.
- Current head dimension limit: `K <= 128`.

Unsupported cases:

- CPU execution.
- Backward/autograd support for `chunk_gated_delta_rule_fwd`.
- `K > 128`.
- Zero-length sequences in `cu_seqlens`.
- Treating returned `A` as a real FLA backward workspace.

`A` is returned only as a shape-compatible forward ABI slot. The current
operator is forward-only, matching `conf/operators.yaml`
`chunk_gated_delta_rule_fwd`, and does not materialize FLA's intra-chunk
solve-triangular workspace.

## Correctness

The new pytest file uses an independent torch recurrence reference. It checks
the output, final state, and local chunk cumsum result. Coverage includes:

- `float16` and `bfloat16`.
- Tiny, medium, and reduced FLA-like shapes.
- `K=128`.
- `beta` as `[B, T, HV]` and `[B, T, HV, V]`.
- `initial_state is None` and provided.
- `output_final_state=True` and `False`.
- `cu_seqlens` variable-length input.
- Non-contiguous q/k/v views.
- Negative guards for invalid heads, invalid `K`, and invalid `cu_seqlens`.

Commands:

```bash
python3 -m py_compile \
  scripts/chunk_gated_delta_rule/inspect_reference.py \
  src/flag_gems/fused/FLA/chunk.py \
  src/flag_gems/fused/FLA/fused_recurrent.py \
  tests/test_FLA/test_chunk_gated_delta_rule.py \
  tests/test_FLA/test_fused_recurrent_gated_delta_rule.py \
  benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py \
  benchmark/test_FLA/test_fused_recurrent_gated_delta_rule_perf.py
bash scripts/chunk_gated_delta_rule/run_accuracy_quick.sh
bash scripts/chunk_gated_delta_rule/run_accuracy_full.sh
pytest tests/test_FLA/test_fused_recurrent_gated_delta_rule.py tests/test_FLA/test_fla_utils_input_guard.py -q
```

Latest results:

```text
py_compile: pass
run_accuracy_quick.sh: 8 passed
run_accuracy_full.sh: 8 passed
pytest tests/test_FLA/test_fused_recurrent_gated_delta_rule.py tests/test_FLA/test_chunk_gated_delta_rule.py -q:
8 passed, 10 skipped
```

## Performance

Command:

```bash
bash scripts/chunk_gated_delta_rule/run_benchmark.sh
```

Latest results on Iluvatar BI-V150 / CoreX:

| Shape | dtype | FlagGems ms | torch ref ms | Speedup |
|---|---:|---:|---:|---:|
| tiny B=1 T=1 H=1 HV=1 K=16 V=16 | fp16 | 0.357 | 0.449 | 1.26x |
| small B=1 T=32 H=4 HV=4 K=64 V=64 | fp16 | 0.354 | 7.978 | 22.55x |
| medium B=1 T=128 H=4 HV=4 K=64 V=64 | fp16 | 0.467 | 31.253 | 66.93x |
| long B=1 T=512 H=4 HV=4 K=64 V=64 | bf16 | 1.381 | 124.049 | 89.85x |
| reduced FLA-like B=1 T=128 H=4 HV=8 K=128 V=128 | bf16 | 0.729 | 31.520 | 43.26x |

## Checklist

- [x] Implementation is scoped to `chunk_gated_delta_rule_fwd`.
- [x] Export path is available through `flag_gems.chunk_gated_delta_rule_fwd`.
- [x] Operator config already contains `chunk_gated_delta_rule_fwd`.
- [x] Correctness tests pass.
- [x] Benchmark passes and includes short and long sequence lengths.
- [x] No external model repository is vendored.
- [x] Unsupported cases and forward-only `A` semantics are documented.

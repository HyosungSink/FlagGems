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
- `cu_seqlens`: `None` or rank-1 `torch.int32` / `torch.int64`; when
  provided, `B == 1`.
- Dtypes: `q/k/v/beta` share `float16`, `bfloat16`, or `float32`; `g` may
  independently use `float16`, `bfloat16`, or `float32`.
- Head grouping: `HV` must be a positive multiple of `H`.
- Current head dimension limit: `K <= 128`.

Unsupported cases:

- CPU execution.
- Backward/autograd support for `chunk_gated_delta_rule_fwd`.
- `K > 128`.
- Zero-length sequences in `cu_seqlens`.
- Treating returned `A` as a backward workspace when `beta` is
  `[B, T, HV, V]`.

Return dtype semantics:

- `g` output keeps `g.dtype`.
- `o` keeps `q/k/v.dtype`.
- `A` keeps `q/k/v.dtype` for scalar beta and is the real FLA/Megatron
  solved intra-chunk workspace `(I + A)^-1`, shape `[B, T, HV, 64]`.
- `A is None` for value-dependent beta `[B, T, HV, V]`, because FLA's scalar
  workspace shape cannot represent per-value beta recurrence without exposing
  a misleading tensor.
- `final_state` is `float32`, matching the internal recurrent accumulator.

The Python wrapper performs one host sync for `cu_seqlens` to build
per-sequence state indices and the torch-side `A` workspace by segment.

## Correctness

The pytest file uses an independent Megatron/FLA torch chunk-formula reference
for scalar beta. It checks `g`, output, real `A`, and final state. The
value-dependent beta extension is checked with an independent torch recurrence
and explicitly verifies `A is None`.

Coverage includes:

- `float16`, `bfloat16`, and `float32`.
- `g` as `float32` with `q/k/v` as `float16` or `bfloat16`.
- Tiny, medium, cross-chunk `T=65` / `T=129`, and reduced FLA-like shapes.
- `K=128`.
- `beta` as `[B, T, HV]` and `[B, T, HV, V]`.
- `initial_state is None` and provided.
- `output_final_state=True` and `False`.
- `cu_seqlens` variable-length input with `torch.int32` and `torch.int64`,
  including sequences crossing the 64-token chunk boundary.
- Non-contiguous q/k/v views.
- Negative guards for invalid heads, invalid `K`, and invalid `cu_seqlens`.
- `SUPPRESS_LEVEL >= 3` raises a clear error because the recurrent
  forward-only path does not materialize `w/h/v_new` debug tensors.

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

Latest local results:

```text
py_compile: pass
pytest tests/test_FLA/test_chunk_gated_delta_rule.py -q -s:
12 passed
pytest tests/test_FLA/test_fused_recurrent_gated_delta_rule.py -q -s:
2 passed, 10 skipped
```

## Performance

Command:

```bash
bash scripts/chunk_gated_delta_rule/run_benchmark.sh
```

Latest results on Iluvatar BI-V150 / CoreX:

The primary benchmark baseline is a Megatron/FLA torch chunk-formula reference.
It is not the earlier per-token Python recurrence baseline.

| Shape | dtype | FlagGems ms | Megatron torch chunk ref ms | Speedup |
|---|---:|---:|---:|---:|
| tiny B=1 T=1 H=1 HV=1 K=16 V=16 | fp16 | 0.647 | 0.836 | 1.29x |
| small B=1 T=32 H=4 HV=4 K=64 V=64 | fp16 | 3.107 | 3.301 | 1.06x |
| cross-chunk B=1 T=65 H=4 HV=4 K=64 V=64 | fp16 | 5.865 | 6.477 | 1.10x |
| medium B=1 T=128 H=4 HV=4 K=64 V=64 | fp16 | 10.678 | 11.432 | 1.07x |
| long B=1 T=512 H=4 HV=4 K=64 V=64 | bf16 | 41.487 | 44.972 | 1.08x |
| reduced FLA-like B=1 T=128 H=4 HV=8 K=128 V=128 | bf16 | 10.783 | 11.527 | 1.07x |

## Checklist

- [x] Implementation is scoped to `chunk_gated_delta_rule_fwd`.
- [x] Export path is available through `flag_gems.chunk_gated_delta_rule_fwd`.
- [x] Operator config already contains `chunk_gated_delta_rule_fwd`.
- [x] Correctness tests pass.
- [x] Benchmark passes and includes short and long sequence lengths.
- [x] No external model repository is vendored.
- [x] Unsupported cases, forward-only scope, and scalar/value-beta `A`
  semantics are documented.

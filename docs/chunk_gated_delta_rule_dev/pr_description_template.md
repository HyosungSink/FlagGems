# [FlagGems Operator Development Competition] Add chunk_gated_delta_rule operator

## Summary

This PR adds forward support for `chunk_gated_delta_rule_fwd` in the existing
FlagGems FLA fused path. On the current CoreX/Iluvatar environment, the public
callable is a **recurrent-backed forward implementation**: it reuses
`fused_recurrent_gated_delta_rule_fwd` for the recurrence and materializes the
FLA/Megatron scalar-beta `A` workspace with a torch formula.

The repository also contains the original fused chunk pipeline
(`fused_cumsum_kkt_solve_tril` + `recompute_w_u` + `chunk_h` + `chunk_o`) behind
`FLAGGEMS_CHUNK_GDR_BACKEND=optimized`, but it is not the default because the
CoreX/Iluvatar Triton compiler aborts while lowering the fused cumsum/KKT/
solve-tril kernel. This PR therefore does **not** claim an optimized FLA chunk
kernel speedup on CoreX.

## Reference

Reference sources inspected:

- FLA pip packages: `flash-linear-attention==0.5.0`, `fla-core==0.5.0`.
- Megatron Core pip package: `megatron-core==0.16.1`.
- The current CoreX image has `torch==2.7.1+corex.4.4.0`.

In this environment, importing `fla.ops.gated_delta_rule.chunk` fails during
Triton/autotuner setup with `ValueError: 'BT' is not in list`. Megatron Core
source is present, but normal module lookup resolves CoreX's preinstalled
`megatron` package first, so `megatron.core.ssm.gated_delta_net` is not
importable. `inspect_reference.py` records these import states and source
fallback locations.

Attempting to run the local optimized fused chunk pipeline on CoreX/Iluvatar
aborts the Python process in Triton lowering with:

```text
TritonILUVATARGPUToLLVM/MemoryOpToLLVM.cpp:52:
Assertion `loadOp && "if use sme/async_cp, LoadOp must be ConvertLayoutOp..."' failed.
Fatal Python error: Aborted
```

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
- `beta`: scalar beta `[B, T, HV]` only.
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
- Value-dependent beta `[B, T, HV, V]`; this raises `NotImplementedError`
  because FLA's `[B, T, HV, 64]` `A` workspace cannot represent per-value beta.
- `SUPPRESS_LEVEL >= 3` on the default recurrent-backed path; debug/recompute
  tensors `w/h/v_new` are only available if the opt-in optimized backend
  compiles.

Return dtype semantics:

- `g` output keeps `g.dtype`.
- `o` keeps `q/k/v.dtype`.
- `A` keeps `q/k/v.dtype` and has shape `[B, T, HV, 64]`.
- `A` is a real scalar-beta FLA/Megatron solved intra-chunk workspace
  `(I + beta*K*K^T*decay)^-1`; on the default path it is materialized by a
  source-derived torch formula, not by the optimized chunk kernel.
- `final_state` is `float32`, matching the internal recurrent accumulator.

The default recurrent-backed path performs one host sync for `cu_seqlens` to
build per-sequence state indices and assemble the torch-side `A` workspace by
segment.

## Correctness

The pytest file uses an independent Megatron/FLA torch chunk-formula reference
for scalar beta. It checks `g`, output, real `A`, and final state. Coverage
includes:

- `float16`, `bfloat16`, and `float32`.
- `g` as `float32` with `q/k/v` as `float16` or `bfloat16`.
- Tiny, medium, cross-chunk `T=65` / `T=129`, and reduced FLA-like shapes.
- `K=128`.
- Scalar `beta` `[B, T, HV]`; 4D beta rejection is tested.
- `initial_state is None` and provided.
- `output_final_state=True` and `False`.
- `cu_seqlens` variable-length input with `torch.int32` and `torch.int64`,
  including sequences crossing the 64-token chunk boundary.
- Non-contiguous q/k/v views.
- Negative guards for invalid heads, invalid `K`, invalid `cu_seqlens`,
  zero-length segments, 4D beta, and default-path `SUPPRESS_LEVEL >= 3`.
- Long-sequence error stats: `T=257 fp16 out_max_abs=0.000702`,
  `A_max_abs=0.000000`, `final_state_max_abs=0.000807`.

Optional optimized-backend A comparison:

- `FLAGGEMS_CHUNK_GDR_RUN_OPTIMIZED_REFERENCE=1` enables a test comparing the
  default path with `FLAGGEMS_CHUNK_GDR_BACKEND=optimized`.
- It is skipped on this CoreX/Iluvatar runner because the optimized kernel
  aborts in Triton lowering.

Latest local results:

```text
git diff --check: pass
py_compile: pass
pytest tests/test_FLA/test_chunk_gated_delta_rule.py -q -s:
13 passed, 1 skipped
pytest tests/test_FLA/test_fused_recurrent_gated_delta_rule.py -q -s:
2 passed, 10 skipped
```

The fused recurrent vLLM reference test now probes the vLLM import in a
subprocess. On this CoreX image it skips the vLLM cases because the probe
aborts on ixformer duplicate native op registration; in an environment where
the same import is safe, the vLLM reference tests run automatically.

## Performance

Command:

```bash
pytest benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py -q -s
```

Latest results on Iluvatar BI-V150 / CoreX:

- `FlagGems ms`: current PR default, recurrent-backed forward plus torch `A`
  materialization.
- `Fused recurrent core ms`: optimized recurrent kernel only, without chunk
  `A` materialization.
- `Megatron torch chunk reference ms`: correctness/slow torch reference
  baseline, not an optimized FLA kernel.
- Optimized FLA/chunk kernel baseline is unavailable on this CoreX runner due
  the Triton lowering abort above.

| Shape | dtype | FlagGems ms | Fused recurrent core ms | Megatron torch chunk reference ms | Torch ref ratio | Overhead vs recurrent |
|---|---:|---:|---:|---:|---:|---:|
| tiny B=1 T=1 H=1 HV=1 K=16 V=16 | fp16 | 0.656 | 0.238 | 0.821 | 1.25x | 2.75x |
| small B=1 T=32 H=4 HV=4 K=64 V=64 | fp16 | 3.092 | 0.268 | 3.315 | 1.07x | 11.53x |
| cross-chunk B=1 T=65 H=4 HV=4 K=64 V=64 | fp16 | 5.877 | 0.294 | 6.516 | 1.11x | 19.97x |
| medium B=1 T=128 H=4 HV=4 K=64 V=64 | fp16 | 10.746 | 0.382 | 11.450 | 1.07x | 28.14x |
| long B=1 T=512 H=4 HV=4 K=64 V=64 | bf16 | 41.558 | 1.227 | 45.028 | 1.08x | 33.88x |
| reduced FLA-like B=1 T=128 H=4 HV=8 K=128 V=128 | bf16 | 10.719 | 0.678 | 11.616 | 1.08x | 15.81x |

These numbers are reported as transparency for the recurrent-backed fallback,
not as evidence of outperforming an optimized FLA chunk kernel.

## Remaining Risks

- The default CoreX path is not a true optimized chunk kernel.
- The opt-in optimized chunk pipeline is present but not validated on this
  CoreX/Iluvatar runner because its Triton kernel aborts at compile/lowering
  time. It should be validated separately on a compatible NVIDIA/Triton stack.
- `A` correctness on CoreX is checked against source-derived Megatron/FLA torch
  formulas; direct optimized FLA A comparison is gated behind the optional
  optimized-reference test.
- The recurrent kernel always needs a final-state workspace internally, so the
  default path cannot fully avoid final-state writes when
  `initial_state is None and output_final_state=False` without a separate
  kernel change.
- The varlen path keeps a `cu_seqlens.detach().cpu().tolist()` host sync.
- Zero-length `cu_seqlens` segments are intentionally rejected.

## Checklist

- [x] Implementation is scoped to `chunk_gated_delta_rule_fwd`.
- [x] Export path is available through `flag_gems.chunk_gated_delta_rule_fwd`.
- [x] Operator config already contains `chunk_gated_delta_rule_fwd`.
- [x] Correctness tests pass on the CoreX recurrent-backed path.
- [x] Benchmark passes and labels baseline types explicitly.
- [x] No external model repository is vendored.
- [x] Unsupported cases and recurrent-backed performance scope are documented.

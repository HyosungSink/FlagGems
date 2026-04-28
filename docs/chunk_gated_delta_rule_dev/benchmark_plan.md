# `chunk_gated_delta_rule` Benchmark Plan

Benchmark file target:

```text
benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py
```

## Baselines

Preferred baselines:

- FLA `chunk_gated_delta_rule`.
- Megatron Core torch-native `torch_chunk_gated_delta_rule`, if available.

If the external reference is unavailable in CI, keep benchmark imports guarded
and fall back cleanly.

## Shapes

Start with:

```text
B=1
T in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
H/HV/K/V small debug set: H=4, HV=4, K=64, V=64
FLA-like set: H=16, HV=32, K=128, V=128
```

Use the same shape builder in tests and benchmarks when possible.

## Metrics

Record:

- Forward latency.
- Speedup versus baseline.
- Dtype.
- Shape.
- Whether final state is requested.
- Whether `cu_seqlens` is used.

## Practical Gates

- Benchmark must not hide small-shape regressions.
- Include at least one realistic long sequence case.
- Keep benchmark generation deterministic.
- Save the exact command and environment used for the PR description.

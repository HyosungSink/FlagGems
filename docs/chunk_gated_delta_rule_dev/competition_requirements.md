# Competition Requirements for `chunk_gated_delta_rule`

Source pages:

- ModelScope Track 1 statement:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- FlagGems pull requests:
  https://github.com/flagos-ai/FlagGems/pulls
- FlagGems operator config:
  `conf/operators.yaml`

## Operator

Competition row:

```text
chunk_gated_delta_rule
Difficulty: advanced
Category: fused
Reference family: Flash Linear Attention / Gated Delta Rule
```

Local operator config currently contains:

```text
id: chunk_gated_delta_rule_fwd
description: The forward case for ChunkGatedDeltaRuleFunction with Flash Linear Attention (FLA).
labels: fused, FLA
kind: Attention
stage: alpha 5.0
exposed: false
```

## Hard Requirements

- Match the selected reference implementation for supported inputs.
- Provide pytest coverage for correctness and edge cases.
- Provide benchmark coverage against the selected reference/baseline.
- Show speedup versus the baseline where the competition requires it.
- Keep implementation scoped to this operator.
- Follow FlagGems style, registration, and testing conventions.
- PR title must contain:

```text
[FlagGems Operator Development Competition]
```

## Practical Interpretation

This is a fused attention-style operator, so correctness evidence matters as
much as raw speed. The PR should make these points obvious:

- Exact input shapes and supported layouts are documented.
- Output and optional final state match the reference within dtype-appropriate tolerances.
- `float16` and `bfloat16` are tested if the reference supports them.
- Variable sequence length behavior is tested if `cu_seqlens` is supported.
- Benchmark shapes include both small and realistic sequence lengths.
- Unsupported cases are explicitly avoided or documented.

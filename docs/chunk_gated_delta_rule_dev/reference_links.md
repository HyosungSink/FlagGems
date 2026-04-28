# Reference Links for `chunk_gated_delta_rule`

## Competition

- ModelScope Track 1:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- Open FlagGems PRs:
  https://github.com/flagos-ai/FlagGems/pulls?q=is%3Apr+chunk_gated_delta_rule

## Primary References

- Flash Linear Attention repository:
  https://github.com/fla-org/flash-linear-attention
- FLA gated delta rule implementation:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
- Megatron Core Gated Delta Net docs:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.ssm.gated_delta_net.html

## Existing FlagGems Patterns

- `conf/operators.yaml`
- `tests/test_FLA/test_fused_recurrent_gated_delta_rule.py`
- `benchmark/test_FLA/test_fused_recurrent_gated_delta_rule_perf.py`
- `src/flag_gems/fused/rwkv_ka_fusion.py`
- `src/flag_gems/fused/rwkv_mm_sparsity.py`
- `tests/test_rwkv_ka_fusion.py`
- `tests/test_rwkv_mm_sparsity.py`

## Public Competition PRs To Monitor

- https://github.com/flagos-ai/FlagGems/pull/2268
- https://github.com/flagos-ai/FlagGems/pull/2067
- https://github.com/flagos-ai/FlagGems/pull/1650

Notes:

- PR 2268 is the closest integration reference, with a small file set and clean CI.
- PR 2067 includes benchmark and fused-path structure ideas.
- PR 1650 is broad and should be used only for algorithmic hints, not as a structural model.

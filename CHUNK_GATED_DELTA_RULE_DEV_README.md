# Chunk Gated Delta Rule Competition Workspace

This repository is a FlagGems development workspace prepared for the
`chunk_gated_delta_rule` operator task in the FlagOS / ModelScope Track 1
competition.

Target operator:

```text
chunk_gated_delta_rule
Difficulty: advanced
Category: fused / FLA attention
Expected public API:
chunk_gated_delta_rule_fwd(q, k, v, g, beta, ...)
```

The task is not a standard PyTorch ATen operator. Treat the public FLA
implementation and any torch-native reference implementation as the semantic
source of truth. On CoreX/Iluvatar, document clearly whether the active path is
the optimized Triton chunk pipeline or a recurrent-backed forward fallback.

## Start Here

- `docs/chunk_gated_delta_rule_dev/competition_requirements.md`
- `docs/chunk_gated_delta_rule_dev/reference_links.md`
- `docs/chunk_gated_delta_rule_dev/implementation_plan.md`
- `docs/chunk_gated_delta_rule_dev/test_matrix.md`
- `docs/chunk_gated_delta_rule_dev/benchmark_plan.md`
- `docs/chunk_gated_delta_rule_dev/pr_checklist.md`

## Cloud Setup

After migrating this directory to the cloud GPU environment, run:

```bash
bash scripts/chunk_gated_delta_rule/setup_cloud_env.sh
```

After implementation and registration, use:

```bash
bash scripts/chunk_gated_delta_rule/run_accuracy_quick.sh
bash scripts/chunk_gated_delta_rule/run_accuracy_full.sh
bash scripts/chunk_gated_delta_rule/run_benchmark.sh
bash scripts/chunk_gated_delta_rule/check_pr_ready.sh
```

## Expected Files To Edit

Likely implementation shape:

- `src/flag_gems/ops/chunk_gated_delta_rule.py` or
  `src/flag_gems/fused/chunk_gated_delta_rule.py`
- `src/flag_gems/ops/__init__.py` or `src/flag_gems/fused/__init__.py`
- `src/flag_gems/__init__.py` if registering a callable override
- `tests/test_chunk_gated_delta_rule.py` or
  `tests/test_FLA/test_chunk_gated_delta_rule.py`
- `benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py`
- `benchmark/core_shapes.yaml` if named shapes are needed

Keep the PR narrow. Do not vendor an entire model repository into FlagGems.

## Competition Notes

The currently strongest public PR for this operator is small and already has a
clean CI surface. To compete, this workspace should focus on one or more of:

- More complete semantic coverage than existing PRs.
- Stronger benchmark evidence across realistic sequence lengths.
- Cleaner integration with the current FlagGems registration style.
- Better maintainability and smaller code footprint.

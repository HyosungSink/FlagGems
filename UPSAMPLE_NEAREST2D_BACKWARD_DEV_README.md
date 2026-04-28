# Upsample Nearest2D Backward Competition Workspace

This repository is a FlagGems development workspace prepared for the
`upsample_nearest2d_backward` part of the FlagOS / ModelScope Track 1 operator
competition.

Target ATen schema:

```text
aten::upsample_nearest2d_backward(
    Tensor grad_output,
    SymInt[2] output_size,
    SymInt[4] input_size,
    float? scales_h=None,
    float? scales_w=None,
) -> Tensor
```

The forward operator is already present in the upstream tree:

```text
src/flag_gems/ops/upsample_nearest2d.py
tests/test_upsample_nearest2d.py
benchmark/test_upsample_nearest2d.py
```

This workspace intentionally does not register a placeholder backward operator.
Register only after the kernel, tests, and benchmark are ready.

## Prepared Materials

- `docs/upsample_nearest2d_backward_dev/competition_requirements.md`
- `docs/upsample_nearest2d_backward_dev/implementation_plan.md`
- `docs/upsample_nearest2d_backward_dev/test_matrix.md`
- `docs/upsample_nearest2d_backward_dev/benchmark_plan.md`
- `docs/upsample_nearest2d_backward_dev/pr_checklist.md`
- `docs/upsample_nearest2d_backward_dev/code_skeleton.md`
- `docs/upsample_nearest2d_backward_dev/reference_links.md`
- `docs/upsample_nearest2d_backward_dev/cloud_migration.md`
- `scripts/upsample_nearest2d_backward/inspect_reference.py`
- `scripts/upsample_nearest2d_backward/setup_cloud_env.sh`
- `scripts/upsample_nearest2d_backward/run_accuracy_quick.sh`
- `scripts/upsample_nearest2d_backward/run_accuracy_full.sh`
- `scripts/upsample_nearest2d_backward/run_benchmark.sh`
- `scripts/upsample_nearest2d_backward/check_pr_ready.sh`

## Quick Start

After moving the repository to a cloud development environment:

```bash
cd upsample_nearest2d_backward_flaggems_dev
bash scripts/upsample_nearest2d_backward/setup_cloud_env.sh
python scripts/upsample_nearest2d_backward/inspect_reference.py --device cuda
```

When the operator is implemented and registered, use:

```bash
bash scripts/upsample_nearest2d_backward/run_accuracy_quick.sh
bash scripts/upsample_nearest2d_backward/run_accuracy_full.sh
bash scripts/upsample_nearest2d_backward/run_benchmark.sh
bash scripts/upsample_nearest2d_backward/check_pr_ready.sh
```

## Expected Implementation Files

- `src/flag_gems/ops/upsample_nearest2d_backward.py`
- `src/flag_gems/ops/__init__.py`
- `src/flag_gems/__init__.py`
- `tests/test_upsample_nearest2d_backward.py`
- `benchmark/test_upsample_nearest2d_backward.py`

## Competition Context

The currently strongest public PR is:

- https://github.com/flagos-ai/FlagGems/pull/2262

It adds a narrow backward implementation and has passed the main Python checks.
A competitive PR should be at least as clean, but stronger on coverage,
benchmark evidence, and edge-case semantics.


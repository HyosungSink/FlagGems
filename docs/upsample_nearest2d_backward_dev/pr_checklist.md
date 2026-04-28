# PR Checklist for `upsample_nearest2d_backward`

## Title

```text
[FlagGems Operator Development Competition] Add upsample_nearest2d_backward operator
```

## Files

- [ ] `src/flag_gems/ops/upsample_nearest2d_backward.py`
- [ ] `src/flag_gems/ops/__init__.py`
- [ ] `src/flag_gems/__init__.py`
- [ ] `tests/test_upsample_nearest2d_backward.py`
- [ ] `benchmark/test_upsample_nearest2d_backward.py`

## Correctness

- [ ] Direct ATen backward comparisons pass.
- [ ] Autograd-through-forward comparisons pass.
- [ ] Identity, integer upsample, fractional upsample, downsample, and singleton
      cases pass.
- [ ] Explicit scale arguments are covered.
- [ ] Dtype coverage matches the repository expectation.

## Performance

- [ ] Benchmark file exists.
- [ ] Benchmark logs include representative scale factors.
- [ ] Weak cases are disclosed or optimized.
- [ ] The PR description includes speedup data.

## Integration

- [ ] Operator is imported in `src/flag_gems/ops/__init__.py`.
- [ ] ATen overload is registered in `src/flag_gems/__init__.py`.
- [ ] No unrelated refactors or bundled external code.
- [ ] Code style passes.

## Local Commands

```bash
bash scripts/upsample_nearest2d_backward/run_accuracy_quick.sh
bash scripts/upsample_nearest2d_backward/run_accuracy_full.sh
bash scripts/upsample_nearest2d_backward/run_benchmark.sh
bash scripts/upsample_nearest2d_backward/check_pr_ready.sh
```


# PR Checklist for `ctc_loss`

PR title:

```text
[FlagGems Operator Development Competition] Add ctc_loss operator
```

## Code

- `src/flag_gems/ops/ctc_loss.py` added.
- Forward and backward implemented.
- Internal math is numerically stable.
- No broad unrelated changes.
- No hard-coded local paths.
- No placeholder registration.
- Style passes pre-commit or project formatting.

## Registration

- Exported from `src/flag_gems/ops/__init__.py`.
- Registered in `src/flag_gems/__init__.py`.
- Uses the correct ATen overload name for `ctc_loss`.

## Correctness

- `pytest tests/test_ctc_loss.py --ref cpu --quick` passes.
- Full `tests/test_ctc_loss.py` passes.
- Forward outputs match PyTorch.
- Backward gradients match PyTorch.
- Both target formats are covered.
- All reductions are covered.
- `zero_infinity` is covered.
- Impossible alignments are covered.
- Non-contiguous cases are covered.

## Performance

- `benchmark/test_ctc_loss.py` exists.
- Forward and backward are measured.
- Device-side speedup is `>= 0.9x` for intended supported cases.
- Minimum speedup is reported, not only representative fast cases.
- Benchmark command and environment are documented.

## PR Description

Include:

- API schema.
- Supported dtypes.
- Supported target layouts.
- Accuracy command and summary.
- Benchmark command and summary table.
- Known limitations, if any.


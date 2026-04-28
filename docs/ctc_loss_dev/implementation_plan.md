# `ctc_loss` Implementation Plan

This plan is intentionally staged. It keeps correctness and performance visible
throughout the work instead of saving benchmark risk for the end.

## Stage 0: Baseline and Shape Harness

Create a small local reference harness before writing Triton kernels:

- Generate valid CTC input cases.
- Compare PyTorch outputs for:
  - padded 2D targets
  - concatenated 1D targets
  - `reduction` in `none`, `mean`, `sum`
  - `zero_infinity` in `False`, `True`
  - `blank` at `0` and non-zero class indices
- Include impossible alignments and empty targets.

Keep this harness separate from the final pytest file until semantics are
stable.

## Stage 1: Forward Only

Implement a forward path first. Suggested starting constraints:

- Contiguous `log_probs` with shape `(T, N, C)` and optional `(T, C)`.
- Padded 2D targets first, then concatenated 1D targets.
- Compute internally in `float32`.
- Produce PyTorch-compatible dtype/output shapes.

CTC forward recurrence:

- Work in log-space.
- Track blank and label states, or an expanded target sequence with blanks.
- Use logaddexp-style accumulation.
- Save only the data needed for backward, because full DP history can be large.

## Stage 2: Backward

Backward is the likely performance bottleneck. Build it early.

Required behavior:

- Support all reductions.
- Respect `zero_infinity`.
- Match PyTorch gradients for `log_probs`.
- Handle repeated labels correctly.
- Handle impossible alignments without NaNs when `zero_infinity=True`.

Implementation hints:

- Reverse-time beta DP is the natural mirror of forward alpha DP.
- Avoid materializing unnecessary full `(T, N, S)` history if it makes memory
  bandwidth dominate.
- Measure backward separately from forward.

## Stage 3: Integration

Only after forward and backward are locally validated:

- Add `src/flag_gems/ops/ctc_loss.py`.
- Export from `src/flag_gems/ops/__init__.py`.
- Register the ATen overload in `src/flag_gems/__init__.py`.
- Add tests in `tests/test_ctc_loss.py`.
- Add benchmark in `benchmark/test_ctc_loss.py`.
- Add shapes in `benchmark/core_shapes.yaml` if the benchmark needs a named
  shape block.

## Stage 4: Competition Proof

Before opening a PR, collect evidence:

- Accuracy logs for quick and full suites.
- Benchmark logs for forward and backward.
- Minimum speedup per dtype and shape group.
- Coverage table mapping test cases to competition requirements.
- Known unsupported cases, if any. Keep this list short.

## Public PR Lessons

Existing public `ctc_loss` PRs show that "mostly fast" may not be enough.
Backward minimum speed can drop below `0.9x` even when forward is strong. Design
the benchmark table so weak cases are visible and fixed, not hidden.


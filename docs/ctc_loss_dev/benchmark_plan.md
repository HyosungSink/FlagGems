# `ctc_loss` Benchmark Plan

The benchmark must prove that the implementation is competitive across forward
and backward. Do not report only the best cases.

## Scripts

Use the repository benchmark framework through:

```bash
bash scripts/ctc_loss/run_benchmark.sh
```

Expected final benchmark file:

```text
benchmark/test_ctc_loss.py
```

Optional shape block in:

```text
benchmark/core_shapes.yaml
```

## Required Comparisons

Compare against:

```python
torch.nn.functional.ctc_loss
```

Measure:

- Forward API latency.
- Backward latency.
- Optional combined forward+backward latency.

Report:

- Minimum speedup.
- Average speedup.
- Maximum speedup.
- Cases below `0.9x`, if any.

## Shape Suggestions

Use CTC-style tuples `(T, N, C, S)`:

```text
small:  (64,  4,  32, 16)
medium: (256, 16, 64, 48)
large:  (512, 32, 64, 48)
stress: (1024, 32, 128, 96)
```

Where:

- `T`: input time steps
- `N`: batch size
- `C`: class count including blank
- `S`: max target length

Include both padded and concatenated targets.

## Dtypes

Required:

- `float32`
- `float16`

Optional:

- `bfloat16`

For dtype limitations in PyTorch native CUDA `ctc_loss`, document whether the
baseline promotes inputs internally. This matters for competition review.

## Minimum Acceptance Gate

Treat the PR as not ready if any intended benchmark category is below `0.9x`.
For `ctc_loss`, backward is the category most likely to fail this gate.


# `upsample_nearest2d_backward` Benchmark Plan

Benchmark file:

```text
benchmark/test_upsample_nearest2d_backward.py
```

Reference operator:

```python
torch.ops.aten.upsample_nearest2d_backward
```

FlagGems operator:

```python
flag_gems.upsample_nearest2d_backward
```

## Shape Groups

Use NCHW shapes:

- small: `(1, 3, 16, 16)`
- batch: `(8, 3, 64, 64)`
- channel-heavy: `(2, 64, 32, 48)`
- image-like: `(4, 32, 128, 128)`
- larger: `(1, 128, 256, 256)` if memory allows

## Scale Groups

- integer upsample: `(2.0, 2.0)`
- asymmetric upsample: `(2.0, 3.0)`
- fractional upsample: `(2.1, 3.7)`
- downsample: `(0.5, 0.5)`
- mixed: `(1.5, 0.75)`

## Metrics

Record:

- mean latency
- speedup versus PyTorch/native baseline
- dtype
- shape
- output size
- explicit scale args or `None`

## Weak Cases to Watch

- Very large upsample factors can make input-parallel accumulation do a lot of
  work per input pixel.
- Downsample can produce many input pixels with zero gradient.
- Non-contiguous `grad_output` can make indexing expensive.

If the generic path is weak, add specialized kernels for common integer factors
such as `2x` and identity.


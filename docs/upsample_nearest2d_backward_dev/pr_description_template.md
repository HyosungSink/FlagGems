# [FlagGems Operator Development Competition] Add upsample_nearest2d_backward operator

## Summary

This PR adds `aten.upsample_nearest2d_backward` support to FlagGems.

Schema:

```text
aten::upsample_nearest2d_backward(
    Tensor grad_output,
    SymInt[2] output_size,
    SymInt[4] input_size,
    float? scales_h=None,
    float? scales_w=None,
) -> Tensor
```

## Implementation

- Kernel strategy:
- Supported layouts:
- Supported dtypes:
- Scale handling:
- Known limitations:

## Correctness

Commands:

```bash
pytest tests/test_upsample_nearest2d_backward.py --ref cpu --quick -m upsample_nearest2d_backward
pytest tests/test_upsample_nearest2d_backward.py -m upsample_nearest2d_backward
```

Coverage:

- [ ] identity
- [ ] integer upsample
- [ ] fractional upsample
- [ ] downsample
- [ ] singleton spatial dimensions
- [ ] explicit scales
- [ ] autograd consistency

## Benchmark

Command:

```bash
pytest benchmark/test_upsample_nearest2d_backward.py -s --level core --record log
```

Results:

| dtype | shape | output_size | scales | PyTorch | FlagGems | speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| | | | | | | |

## Checklist

- [ ] Tests added.
- [ ] Benchmark added.
- [ ] Registration added.
- [ ] No unrelated files changed.
- [ ] Local checks pass.


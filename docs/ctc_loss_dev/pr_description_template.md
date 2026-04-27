# [FlagGems Operator Development Competition] Add ctc_loss operator

## Summary

This PR adds `ctc_loss` support to FlagGems.

API:

```python
torch.nn.functional.ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
)
```

## Supported Features

- Forward:
- Backward:
- Target layouts:
- Dtypes:
- Reductions:
- `zero_infinity`:
- Non-contiguous inputs:

## Accuracy

Command:

```bash
pytest tests/test_ctc_loss.py --ref cpu --record log
```

Summary:

```text
TODO: paste pass count and notable edge cases.
```

## Benchmark

Command:

```bash
pytest benchmark/test_ctc_loss.py -s --level comprehensive --warmup 20 --iter 50 --record log
```

Summary:

| Direction | DType | Min speedup | Avg speedup | Max speedup |
| --- | --- | ---: | ---: | ---: |
| forward | float32 | TODO | TODO | TODO |
| backward | float32 | TODO | TODO | TODO |
| forward | float16 | TODO | TODO | TODO |
| backward | float16 | TODO | TODO | TODO |

## Competition Checklist

- [ ] Correctness matches PyTorch for covered cases.
- [ ] Forward and backward covered by tests.
- [ ] Benchmark uses FlagGems framework.
- [ ] Minimum device speedup is at least `0.9x`.
- [ ] Code is formatted and scoped to `ctc_loss`.


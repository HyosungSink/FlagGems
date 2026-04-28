# Competition Requirements for `ctc_loss`

Source pages:

- ModelScope Track 1 statement:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- FlagGems pull requests:
  https://github.com/flagos-ai/FlagGems/pulls
- PyTorch reference API:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html

## Operator

Competition row:

```text
ctc_loss
Difficulty: advanced
Category: loss
Torch API:
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

## Hard Requirements

- Match PyTorch semantics for supported inputs.
- Provide complete pytest coverage.
- Use the FlagGems benchmark framework for device-side performance testing.
- Show speedup versus PyTorch/native baseline. Competition text says the
  device-side speedup must be at least `0.9x`.
- At minimum support `float32` and `float16`.
- Follow FlagGems repository structure and style.
- PR title must contain:

```text
[FlagGems Operator Development Competition]
```

## Scoring Dimensions

The competition scores:

- Correctness: 30%
- Performance: 20%
- Open-source integration: 10%
- Cross-platform compatibility: 10%
- Test completeness: 20%
- Readability: 10%

## Practical Interpretation

Passing GitHub checks is not enough for this task. For a competition PR, the
maintainers must be able to conclude that the implementation is the winning
solution for the operator. That means:

- Forward and backward behavior are both correct.
- Edge cases are covered and documented.
- Performance results include weak cases, not only representative fast cases.
- Benchmark commands and logs are reproducible.
- The implementation is small enough and integrated enough to be maintainable.


# Competition Requirements for `upsample_nearest2d_backward`

Source pages:

- ModelScope Track 1 statement:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- FlagGems pull requests:
  https://github.com/flagos-ai/FlagGems/pulls
- PyTorch ATen schema:
  `torch.ops.aten.upsample_nearest2d_backward`

## Operator Row

Competition row:

```text
upsample_nearest2d
Difficulty: medium
Category: upsampling
Schema:
upsample_nearest2d(
    Tensor self,
    SymInt[2] output_size,
    float? scales_h=None,
    float? scales_w=None,
) -> Tensor
upsample_nearest2d_backward(
    Tensor grad_output,
    SymInt[2] output_size,
    SymInt[4] input_size,
    float? scales_h=None,
    float? scales_w=None,
) -> Tensor
```

The upstream repository already has the forward operator. This workspace focuses
on the backward operator.

## Hard Requirements

- Match PyTorch semantics for `aten.upsample_nearest2d_backward`.
- Support 4D NCHW tensors.
- Support `scales_h=None/scales_w=None` and explicit scale values.
- Cover integer upsample, fractional upsample, downsample, identity, and
  singleton spatial dimensions.
- Use FlagGems tests and benchmark style.
- Show performance versus the PyTorch/native baseline.
- Keep the implementation scoped to this operator.
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

Passing CI is necessary but not sufficient. A strong PR should show:

- Accuracy against PyTorch direct backward and autograd-backed forward.
- Coverage for tricky nearest-neighbor index boundaries.
- Benchmark logs across representative shapes and scale factors.
- A clear explanation of whether the kernel uses input-parallel accumulation,
  output-parallel atomic adds, or a hybrid strategy.


# `upsample_nearest2d_backward` Test Matrix

Use this as the coverage checklist for
`tests/test_upsample_nearest2d_backward.py`.

## Shape Coverage

- `(1, 1, 1, 1)` to `(1, 1, 1, 1)`
- `(1, 1, 1, 7)` with width upsample/downsample
- `(1, 1, 7, 1)` with height upsample/downsample
- `(2, 3, 4, 5)` common small case
- `(4, 16, 16, 16)` medium case
- `(2, 64, 32, 48)` channel-heavy case
- `(1, 128, 64, 64)` larger case if memory allows

## Scale Coverage

- identity: `(OH, OW) == (IH, IW)`
- integer upsample: `(2.0, 2.0)`, `(3.0, 2.0)`
- fractional upsample: `(2.1, 3.7)`, `(1.3, 5.1)`
- downsample: `(0.5, 0.5)`, `(0.3, 0.7)`
- mixed: height upsample + width downsample

## Argument Coverage

- `scales_h=None`, `scales_w=None`
- explicit `scales_h`, `scales_w`
- `output_size` as list and tuple if the repo style allows both
- float dtypes from `tests.accuracy_utils.FLOAT_DTYPES`

## Correctness Checks

- Compare direct ATen backward output.
- Compare autograd gradient from forward nearest2d.
- Ensure finite output.
- Ensure shape equals `input_size`.
- Tight tolerance for `float32`; dtype-appropriate tolerance for `float16` and
  `bfloat16`.

## Layout Checks

- Contiguous `grad_output`.
- Non-contiguous `grad_output`, if supported.

If non-contiguous input is intentionally copied, add a focused test proving the
result still matches PyTorch.


# `ctc_loss` Test Matrix

Use this as the coverage checklist for `tests/test_ctc_loss.py`.

## Core Dimensions

Shapes:

- Batched: `log_probs` shape `(T, N, C)`
- Unbatched: `log_probs` shape `(T, C)`
- Small: short `T`, small batch, small class count
- Medium: realistic speech-like values
- Large: stress memory and bandwidth

Targets:

- Padded 2D target tensor: `(N, S)`
- Concatenated 1D target tensor: `(sum(target_lengths),)`
- Empty target for at least one sample
- Repeated labels, for example `[1, 1, 2]`
- Maximum target length close to input length

Parameters:

- `blank=0`
- `blank != 0`
- `reduction="none"`
- `reduction="mean"`
- `reduction="sum"`
- `zero_infinity=False`
- `zero_infinity=True`

Dtypes:

- `torch.float32`
- `torch.float16`
- Optional: `torch.bfloat16`

Layouts:

- Contiguous `log_probs`
- Non-contiguous `log_probs`
- CPU reference path via `--ref cpu`

Edge cases:

- Impossible alignment: `input_length < target_length`
- Impossible repeated-label alignment: repeated labels require extra blanks
- `-inf` values in `log_probs`
- Varying `input_lengths` in the same batch
- Varying `target_lengths` in the same batch
- Invalid target layout should raise or match PyTorch failure behavior

## Suggested Quick Mode

Quick mode should finish fast and still catch semantic regressions:

- dtype: `float32`
- reductions: `mean`, `none`
- target layouts: padded and concatenated
- one impossible alignment with `zero_infinity=True`
- one backward gradient check

## Suggested Full Mode

Full mode should cross product the important dimensions without exploding:

- dtypes: `float32`, `float16`
- reductions: `none`, `mean`, `sum`
- target layouts: padded and concatenated
- zero infinity: both values
- target patterns: normal, repeated label, empty, impossible
- layout: contiguous and representative non-contiguous

## Assertion Strategy

Use the repository helpers in `tests/accuracy_utils.py` where possible.

For `float32`, expect a tight tolerance close to the competition table.
For `float16`, compare against a reference computed in a stable dtype when
PyTorch native support is limited, but document the baseline clearly.


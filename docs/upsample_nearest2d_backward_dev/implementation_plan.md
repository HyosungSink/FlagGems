# `upsample_nearest2d_backward` Implementation Plan

## Stage 0: Reference Harness

Before writing kernels:

- Run `scripts/upsample_nearest2d_backward/inspect_reference.py`.
- Confirm the exact PyTorch schema in the cloud environment.
- Compare direct ATen backward with autograd through forward for:
  - integer upsample
  - fractional upsample
  - downsample
  - identity
  - singleton height/width
  - explicit `scales_h` / `scales_w`

## Stage 1: Minimal Correct Kernel

Implement `src/flag_gems/ops/upsample_nearest2d_backward.py`.

Recommended first version:

- Input-parallel kernel: one program computes a tile of `grad_input`.
- For each input pixel `(ih, iw)`, accumulate output pixels `(oh, ow)` whose
  nearest-neighbor forward map selects `(ih, iw)`.
- Accumulate in `float32`, cast to output dtype at store.
- Support contiguous NCHW first.
- Use `input_size` to allocate `grad_input` with shape `(N, C, IH, IW)`.

This avoids atomics and is easiest to reason about. If performance is weak for
large upsample factors, add specialized paths later.

## Stage 2: Semantics and Edge Cases

Handle:

- `scales_h` and `scales_w` explicitly.
- `output_size` not equal to `input_size * integer_factor`.
- `OH < IH` or `OW < IW`.
- `IH == 1`, `IW == 1`, `OH == 1`, `OW == 1`.
- `grad_output` dtype in the repo's float dtype set.
- Non-contiguous `grad_output` if it can be supported cleanly; otherwise make a
  deliberate `contiguous()` copy and document it in the PR.

## Stage 3: Registration

Only after local correctness is established:

- Import the function from `src/flag_gems/ops/__init__.py`.
- Register it in `src/flag_gems/__init__.py` using the ATen name:

```text
("upsample_nearest2d_backward", upsample_nearest2d_backward)
```

## Stage 4: Tests

Add `tests/test_upsample_nearest2d_backward.py`.

Use both comparison styles:

- Direct PyTorch operator:
  `torch.ops.aten.upsample_nearest2d_backward`
- Autograd consistency:
  run forward `torch._C._nn.upsample_nearest2d`, call backward, compare
  `input.grad`.

## Stage 5: Benchmark Evidence

Add `benchmark/test_upsample_nearest2d_backward.py`.

Benchmark:

- small, medium, large NCHW shapes
- integer scale `2x`
- fractional scale
- downsample
- channel-heavy shapes

Keep logs from:

```bash
bash scripts/upsample_nearest2d_backward/run_benchmark.sh
```


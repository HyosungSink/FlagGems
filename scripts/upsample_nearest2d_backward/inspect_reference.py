#!/usr/bin/env python
"""PyTorch-only reference harness for upsample_nearest2d_backward.

This script does not depend on FlagGems registration. Use it to confirm that a
cloud environment can run the reference cases before implementing kernels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    name: str
    input_size: tuple[int, int, int, int]
    scale: tuple[float, float] | None
    output_size: tuple[int, int] | None = None


def resolve_output_size(case: Case) -> tuple[int, int]:
    if case.output_size is not None:
        return case.output_size
    assert case.scale is not None
    _, _, h, w = case.input_size
    return max(int(h * case.scale[0]), 1), max(int(w * case.scale[1]), 1)


def run_case(case: Case, dtype: torch.dtype, device: torch.device) -> None:
    output_size = resolve_output_size(case)
    scales_h = None if case.scale is None else float(case.scale[0])
    scales_w = None if case.scale is None else float(case.scale[1])

    x = torch.randn(case.input_size, device=device, dtype=dtype, requires_grad=True)
    y = torch.ops.aten.upsample_nearest2d.default(
        x, output_size, scales_h, scales_w
    )
    grad_output = torch.randn_like(y)
    y.backward(grad_output)
    autograd_grad = x.grad.detach()

    direct_grad = torch.ops.aten.upsample_nearest2d_backward.default(
        grad_output, output_size, case.input_size, scales_h, scales_w
    )

    torch.testing.assert_close(direct_grad, autograd_grad, rtol=0, atol=0)
    print(
        case.name,
        "input=",
        case.input_size,
        "output=",
        output_size,
        "dtype=",
        str(dtype).replace("torch.", ""),
        "sum=",
        float(direct_grad.float().sum().item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--all-dtypes", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("torch:", torch.__version__)
    print("device:", device)
    print("forward schema:", torch.ops.aten.upsample_nearest2d.default._schema)
    print(
        "backward schema:",
        torch.ops.aten.upsample_nearest2d_backward.default._schema,
    )

    cases = [
        Case("identity", (1, 1, 4, 5), None, (4, 5)),
        Case("integer_2x", (2, 3, 4, 5), (2.0, 2.0)),
        Case("fractional_up", (2, 3, 4, 5), (2.1, 3.7)),
        Case("downsample", (2, 3, 8, 9), (0.5, 0.5)),
        Case("mixed_scale", (1, 4, 7, 11), (1.5, 0.75)),
        Case("singleton_h", (1, 2, 1, 7), (3.0, 2.0)),
        Case("singleton_w", (1, 2, 7, 1), (2.0, 3.0)),
    ]

    dtypes = [torch.float32]
    if args.all_dtypes and device.type == "cuda":
        dtypes.extend([torch.float16, torch.bfloat16])

    for dtype in dtypes:
        for case in cases:
            run_case(case, dtype, device)


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""Small PyTorch-only sanity harness for CTC loss semantics.

This script does not depend on FlagGems registration. Use it to confirm that a
cloud environment can run the reference cases before implementing kernels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, int, int, int]
    target_layout: str
    reduction: str
    zero_infinity: bool
    dtype: torch.dtype


def make_case(case: Case, device: torch.device):
    t_steps, batch, classes, max_target = case.shape
    raw = torch.randn(t_steps, batch, classes, device=device, dtype=torch.float32)
    log_probs = raw.log_softmax(dim=-1).to(case.dtype).requires_grad_(True)

    target_lengths = torch.randint(
        low=max(1, max_target // 2),
        high=max_target + 1,
        size=(batch,),
        device=device,
        dtype=torch.long,
    )
    input_lengths = torch.full(
        (batch,), t_steps, device=device, dtype=torch.long
    )

    padded = torch.zeros(batch, max_target, device=device, dtype=torch.long)
    pieces = []
    for row, length in enumerate(target_lengths.tolist()):
        vals = torch.randint(1, classes, (length,), device=device, dtype=torch.long)
        padded[row, :length] = vals
        pieces.append(vals)

    if case.target_layout == "padded":
        targets = padded
    elif case.target_layout == "concatenated":
        targets = torch.cat(pieces)
    else:
        raise ValueError(f"unknown target layout: {case.target_layout}")

    return log_probs, targets, input_lengths, target_lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cases = [
        Case("small_padded_mean", (32, 4, 16, 8), "padded", "mean", False, torch.float32),
        Case("small_concat_sum", (32, 4, 16, 8), "concatenated", "sum", False, torch.float32),
        Case("small_padded_none_zi", (32, 4, 16, 8), "padded", "none", True, torch.float32),
    ]

    for case in cases:
        log_probs, targets, input_lengths, target_lengths = make_case(case, device)
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction=case.reduction,
            zero_infinity=case.zero_infinity,
        )
        grad = torch.ones_like(loss) if loss.ndim else None
        loss.backward(grad)
        print(
            case.name,
            "loss_shape=",
            tuple(loss.shape),
            "grad_finite=",
            bool(torch.isfinite(log_probs.grad).all().item()),
        )


if __name__ == "__main__":
    main()


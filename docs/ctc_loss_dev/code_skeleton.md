# Code Skeleton Notes

Do not paste this into registration until the implementation is real. The goal
is to keep the cloud workspace ready without making the current repository
fail imports.

Expected file:

```text
src/flag_gems/ops/ctc_loss.py
```

Suggested structure:

```python
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _ctc_loss_forward_kernel(...):
    ...


@libentry()
@triton.jit
def _ctc_loss_backward_kernel(...):
    ...


class CtcLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths,
                blank=0, reduction="mean", zero_infinity=False):
        ...

    @staticmethod
    def backward(ctx, grad_out):
        ...


def ctc_loss(log_probs, targets, input_lengths, target_lengths,
             blank=0, reduction="mean", zero_infinity=False):
    logger.debug("GEMS CTC LOSS")
    return CtcLossFunction.apply(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )
```

Registration likely belongs in:

```text
src/flag_gems/ops/__init__.py
src/flag_gems/__init__.py
```

Find the exact registration pattern by comparing loss operators already in the
repository, especially:

```text
src/flag_gems/ops/mse_loss.py
src/flag_gems/ops/nllloss.py
src/flag_gems/ops/nll_loss_nd.py
```


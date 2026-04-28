# Code Skeleton for `upsample_nearest2d_backward`

Suggested file:

```text
src/flag_gems/ops/upsample_nearest2d_backward.py
```

Sketch:

```python
import logging
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.jit
def upsample_nearest2d_backward_kernel(...):
    # Input-parallel accumulation:
    #   for each grad_input pixel, sum grad_output pixels whose nearest source
    #   index maps to that input pixel.
    pass


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4
    assert len(output_size) == 2
    assert len(input_size) == 4
    ...
```

Registration after correctness is proven:

```python
# src/flag_gems/ops/__init__.py
from flag_gems.ops.upsample_nearest2d_backward import upsample_nearest2d_backward
```

```python
# src/flag_gems/__init__.py
("upsample_nearest2d_backward", upsample_nearest2d_backward),
```

Do not copy a large external project into the PR. Keep the patch limited to the
operator, tests, benchmark, and minimal tuning config if needed.


# Code Skeleton for `chunk_gated_delta_rule`

Possible implementation file:

```text
src/flag_gems/ops/chunk_gated_delta_rule.py
```

or:

```text
src/flag_gems/fused/chunk_gated_delta_rule.py
```

Suggested Python shape:

```python
import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _chunk_gated_delta_rule_fwd_kernel(...):
    ...


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    logger.debug("GEMS CHUNK GATED DELTA RULE")
    # Validate ranks, dtype, device, and supported options.
    # Allocate output and optional final state.
    # Launch Triton kernel(s).
    # Return output or (output, final_state), matching the chosen reference.
    ...
```

Registration should follow the local FlagGems style already used by fused
operators and competition PRs. Avoid adding a placeholder registration before
the implementation is correct.

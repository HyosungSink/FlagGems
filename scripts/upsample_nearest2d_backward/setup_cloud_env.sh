#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[upsample_nearest2d_backward] repository: ${ROOT_DIR}"
echo "[upsample_nearest2d_backward] branch: $(git branch --show-current 2>/dev/null || echo unknown)"

python - <<'PY'
import importlib.util
import platform
import sys

print("python:", sys.version.replace("\n", " "))
print("platform:", platform.platform())

for name in ["torch", "triton", "pytest", "yaml"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", "found" if spec else "missing")

try:
    import torch

    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    print("aten backward schema:", torch.ops.aten.upsample_nearest2d_backward.default._schema)
except Exception as exc:
    print("torch import error:", repr(exc))
PY

cat <<'EOF'

[upsample_nearest2d_backward] Optional install commands:

  # In a prepared FlagOS cloud image, you may not need this.
  python -m pip install -e ".[test]"

  # NVIDIA extra, only when the package index is accessible from the cloud:
  python -m pip install -e ".[nvidia,test]"

[upsample_nearest2d_backward] Next:

  Read UPSAMPLE_NEAREST2D_BACKWARD_DEV_README.md
  Run: python scripts/upsample_nearest2d_backward/inspect_reference.py --device cuda
  Implement src/flag_gems/ops/upsample_nearest2d_backward.py
  Add tests/test_upsample_nearest2d_backward.py
  Add benchmark/test_upsample_nearest2d_backward.py
EOF


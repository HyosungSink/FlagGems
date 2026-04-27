#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[ctc_loss] repository: ${ROOT_DIR}"
echo "[ctc_loss] branch: $(git branch --show-current 2>/dev/null || echo unknown)"

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
except Exception as exc:
    print("torch import error:", repr(exc))
PY

cat <<'EOF'

[ctc_loss] Optional install commands:

  # In a prepared FlagOS cloud image, you may not need this.
  python -m pip install -e ".[test]"

  # NVIDIA extra, only when the package index is accessible from the cloud:
  python -m pip install -e ".[nvidia,test]"

[ctc_loss] Next:

  Read CTC_LOSS_DEV_README.md
  Implement src/flag_gems/ops/ctc_loss.py
  Add tests/test_ctc_loss.py
  Add benchmark/test_ctc_loss.py
EOF


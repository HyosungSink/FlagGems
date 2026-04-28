#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install -e .
python -m pip install pytest packaging

cat <<'MSG'

Base FlagGems package installed in editable mode.

Optional reference packages for chunk_gated_delta_rule:
  python -m pip install flash-linear-attention
  python -m pip install megatron-core

Install the reference that matches your cloud CUDA/PyTorch image, then run:
  python scripts/chunk_gated_delta_rule/inspect_reference.py

MSG

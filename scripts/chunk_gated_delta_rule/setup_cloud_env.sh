#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

"${PYTHON}" -m pip install -U pip
"${PYTHON}" -m pip install -e .
"${PYTHON}" -m pip install pytest packaging

cat <<MSG

Base FlagGems package installed in editable mode.

Optional reference packages for chunk_gated_delta_rule:
  ${PYTHON} -m pip install flash-linear-attention
  ${PYTHON} -m pip install megatron-core

Install the reference that matches your cloud CUDA/PyTorch image, then run:
  ${PYTHON} scripts/chunk_gated_delta_rule/inspect_reference.py

MSG

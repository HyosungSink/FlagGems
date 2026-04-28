#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

"${PYTHON}" scripts/chunk_gated_delta_rule/inspect_reference.py
bash scripts/chunk_gated_delta_rule/run_accuracy_quick.sh
bash scripts/chunk_gated_delta_rule/run_accuracy_full.sh
bash scripts/chunk_gated_delta_rule/run_benchmark.sh

git status --short

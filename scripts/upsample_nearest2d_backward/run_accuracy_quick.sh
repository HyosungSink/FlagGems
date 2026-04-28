#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TEST_FILE="${UPSAMPLE_NEAREST2D_BACKWARD_TEST_FILE:-tests/test_upsample_nearest2d_backward.py}"

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[upsample_nearest2d_backward] ${TEST_FILE} does not exist yet."
  echo "[upsample_nearest2d_backward] Create it from docs/upsample_nearest2d_backward_dev/test_matrix.md before running accuracy."
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python -m pytest "${TEST_FILE}" --ref cpu --quick -m upsample_nearest2d_backward "$@"


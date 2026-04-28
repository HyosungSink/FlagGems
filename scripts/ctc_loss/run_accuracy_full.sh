#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON:-python3}"

TEST_FILE="${CTC_TEST_FILE:-tests/test_ctc_loss.py}"

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[ctc_loss] ${TEST_FILE} does not exist yet."
  echo "[ctc_loss] Create it from docs/ctc_loss_dev/test_matrix.md before running accuracy."
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m pytest "${TEST_FILE}" --ref cpu --record log -m ctc_loss "$@"

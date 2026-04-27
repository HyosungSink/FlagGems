#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

missing=0

require_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    echo "[ok] ${path}"
  else
    echo "[missing] ${path}"
    missing=1
  fi
}

echo "[ctc_loss] Checking expected competition files"
require_file "src/flag_gems/ops/ctc_loss.py"
require_file "tests/test_ctc_loss.py"
require_file "benchmark/test_ctc_loss.py"

echo
echo "[ctc_loss] Checking registration hints"
if rg -n "ctc_loss" src/flag_gems/ops/__init__.py >/dev/null; then
  echo "[ok] src/flag_gems/ops/__init__.py references ctc_loss"
else
  echo "[missing] src/flag_gems/ops/__init__.py does not reference ctc_loss"
  missing=1
fi

if rg -n '"ctc_loss' src/flag_gems/__init__.py >/dev/null; then
  echo "[ok] src/flag_gems/__init__.py registers ctc_loss"
else
  echo "[missing] src/flag_gems/__init__.py does not register ctc_loss"
  missing=1
fi

echo
echo "[ctc_loss] Git status"
git status --short

if [[ "${missing}" -ne 0 ]]; then
  echo
  echo "[ctc_loss] Not PR-ready yet. See docs/ctc_loss_dev/pr_checklist.md."
  exit 2
fi

echo
echo "[ctc_loss] File-level checklist is satisfied. Run accuracy and benchmark next."


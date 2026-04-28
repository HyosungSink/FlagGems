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

echo "[upsample_nearest2d_backward] Checking expected competition files"
require_file "src/flag_gems/ops/upsample_nearest2d_backward.py"
require_file "tests/test_upsample_nearest2d_backward.py"
require_file "benchmark/test_upsample_nearest2d_backward.py"

echo
echo "[upsample_nearest2d_backward] Checking registration hints"
if rg -n "upsample_nearest2d_backward" src/flag_gems/ops/__init__.py >/dev/null; then
  echo "[ok] src/flag_gems/ops/__init__.py references upsample_nearest2d_backward"
else
  echo "[missing] src/flag_gems/ops/__init__.py does not reference upsample_nearest2d_backward"
  missing=1
fi

if rg -n '"upsample_nearest2d_backward"' src/flag_gems/__init__.py >/dev/null; then
  echo "[ok] src/flag_gems/__init__.py registers upsample_nearest2d_backward"
else
  echo "[missing] src/flag_gems/__init__.py does not register upsample_nearest2d_backward"
  missing=1
fi

echo
echo "[upsample_nearest2d_backward] Git status"
git status --short

if [[ "${missing}" -ne 0 ]]; then
  echo
  echo "[upsample_nearest2d_backward] Not PR-ready yet. See docs/upsample_nearest2d_backward_dev/pr_checklist.md."
  exit 2
fi

echo
echo "[upsample_nearest2d_backward] File-level checklist is satisfied. Run accuracy and benchmark next."


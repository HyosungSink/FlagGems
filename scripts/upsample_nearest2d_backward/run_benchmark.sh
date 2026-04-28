#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BENCH_FILE="${UPSAMPLE_NEAREST2D_BACKWARD_BENCH_FILE:-benchmark/test_upsample_nearest2d_backward.py}"

if [[ ! -f "${BENCH_FILE}" ]]; then
  echo "[upsample_nearest2d_backward] ${BENCH_FILE} does not exist yet."
  echo "[upsample_nearest2d_backward] Create it from docs/upsample_nearest2d_backward_dev/benchmark_plan.md before running benchmark."
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python -m pytest "${BENCH_FILE}" -s --level core --record log "$@"


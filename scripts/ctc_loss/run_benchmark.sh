#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BENCH_FILE="${CTC_BENCH_FILE:-benchmark/test_ctc_loss.py}"
WARMUP="${CTC_WARMUP:-20}"
ITER="${CTC_ITER:-50}"
LEVEL="${CTC_BENCH_LEVEL:-comprehensive}"
DTYPES="${CTC_DTYPES:-}"

if [[ ! -f "${BENCH_FILE}" ]]; then
  echo "[ctc_loss] ${BENCH_FILE} does not exist yet."
  echo "[ctc_loss] Create it from docs/ctc_loss_dev/benchmark_plan.md before running benchmark."
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

args=(
  "${BENCH_FILE}"
  -s
  --level "${LEVEL}"
  --warmup "${WARMUP}"
  --iter "${ITER}"
  --record log
  -m ctc_loss
)

if [[ -n "${DTYPES}" ]]; then
  args+=(--dtypes "${DTYPES}")
fi

python -m pytest "${args[@]}" "$@"


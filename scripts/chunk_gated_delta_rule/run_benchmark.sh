#!/usr/bin/env bash
set -euo pipefail

if [[ -f benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py ]]; then
  pytest benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py -s --record log
elif [[ -f benchmark/test_chunk_gated_delta_rule.py ]]; then
  pytest benchmark/test_chunk_gated_delta_rule.py -s --record log
else
  echo "No chunk_gated_delta_rule benchmark file exists yet."
  echo "Expected benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py or benchmark/test_chunk_gated_delta_rule.py"
  exit 1
fi

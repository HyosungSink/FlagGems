#!/usr/bin/env bash
set -euo pipefail

if [[ -f tests/test_chunk_gated_delta_rule.py ]]; then
  pytest tests/test_chunk_gated_delta_rule.py -q
elif [[ -f tests/test_FLA/test_chunk_gated_delta_rule.py ]]; then
  pytest tests/test_FLA/test_chunk_gated_delta_rule.py -q
else
  echo "No chunk_gated_delta_rule test file exists yet."
  echo "Expected tests/test_chunk_gated_delta_rule.py or tests/test_FLA/test_chunk_gated_delta_rule.py"
  exit 1
fi

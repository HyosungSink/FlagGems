#!/usr/bin/env bash
set -euo pipefail

pytest -q -m "chunk_gated_delta_rule"

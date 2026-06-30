#!/usr/bin/env bash
# Reproduce baseline evaluation. Seed/config overridable so the instructor can
# swap in the held-out stress config and grading seeds.
set -euo pipefail
CONFIG="${1:-configs/eval_standard.yaml}"
SEEDS="${2:-0,1,2,3,4}"
POLICY="${3:-greedy_nearest}"
python -m drone_dispatch_env.run_eval --config "$CONFIG" --seeds "$SEEDS" --policy "$POLICY"

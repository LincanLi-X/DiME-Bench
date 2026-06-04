#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_track4_reasoning_datasets.py --dataset path_star --limit 2
python scripts/run_track4_reasoning.py \
  --model LLaDA-8B \
  --dataset path_star \
  --limit 2 \
  --sample-size 2 \
  --sample-seed 42 \
  --mock-model \
  --no-require-cuda \
  --overwrite

#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_track6_efficiency_datasets.py \
  --dataset all \
  --limit 2 \
  --allow-synthetic-fallback

python scripts/run_track6_efficiency.py \
  --model LLaDA-8B \
  --dataset fixed_length \
  --sweep t \
  --limit 2 \
  --mock-model \
  --no-require-cuda \
  --overwrite

python scripts/run_track6_efficiency.py \
  --model GPT-2-XL \
  --dataset fixed_length \
  --sweep t \
  --limit 2 \
  --dry-run \
  --no-require-cuda \
  --overwrite

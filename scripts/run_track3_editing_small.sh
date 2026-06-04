#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_track3_editing_datasets.py --dataset synthetic_contradiction_repair --limit 2
python scripts/run_track3_editing.py \
  --model GPT-2-XL \
  --dataset synthetic_contradiction_repair \
  --limit 2 \
  --sample-size 2 \
  --sample-seed 42 \
  --mock-model \
  --no-require-cuda \
  --overwrite

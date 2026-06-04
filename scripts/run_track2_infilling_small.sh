#!/usr/bin/env bash
set -euo pipefail

python scripts/check_track2_infilling_setup.py \
  --allow-missing-models \
  --allow-missing-datasets \
  --allow-no-cuda

if [ ! -f data/processed/track2_infilling/wikitext103_sentence/samples.jsonl ]; then
  python scripts/prepare_track2_infilling_datasets.py \
    --dataset wikitext103_sentence \
    --limit 2 \
    --max-source-rows 2000
fi

python scripts/run_track2_infilling.py \
  --model GPT-2-XL \
  --dataset wikitext103_sentence \
  --limit 2 \
  --sample-size 2 \
  --sample-seed 42 \
  --mock-model \
  --overwrite \
  --no-require-cuda

#!/usr/bin/env bash
set -euo pipefail

# Keep CUDA device ids aligned with nvidia-smi physical GPU ids.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# Default to the A100-80GB in the current server nvidia-smi listing.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HF_HOME="${HF_HOME:-$PWD/data/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/data/hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/data/hf_cache/datasets}"

MODELS=(
  "LLaDA-8B"
  "LLaDA-1.5"
  "DREAM-7B"
  "DiffuLLaMA-7B"
  "LLaMA-3-8B-Instruct"
  "GPT-2-XL"
)

DATASETS=(
  "mmlu_pro"
  "gsm8k"
  "humaneval"
  "hellaswag"
  "ifeval"
)

python scripts/prepare_datasets.py --track track1_general

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    python scripts/run_track.py \
      --track track1_general \
      --model "$model" \
      --dataset "$dataset" \
      --overwrite
  done
done

python scripts/aggregate_results.py --track track1_general

#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export HF_HOME="${HF_HOME:-$PWD/data/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/data/hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/data/hf_cache/datasets}"

DATASETS=(
  "mmlu_pro"
  "gsm8k"
  "humaneval"
  "hellaswag"
  "ifeval"
)

run_model() {
  local gpu="$1"
  local model="$2"
  {
    echo "[$(date)] starting ${model} on physical GPU ${gpu}"
    for dataset in "${DATASETS[@]}"; do
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_track.py \
        --track track1_general \
        --model "$model" \
        --dataset "$dataset" \
        --overwrite
    done
    echo "[$(date)] finished ${model}"
  } > "logs/track1_general/${model// /_}.run.log" 2>&1
}

python scripts/prepare_datasets.py --track track1_general
python scripts/check_track1_setup.py

# Avoid GPU 0 because it is occupied by a VLLM process in the current server snapshot.
run_model 3 "LLaDA-8B" &
run_model 1 "LLaDA-1.5" &
run_model 4 "DREAM-7B" &
run_model 2 "DiffuLLaMA-7B" &
wait

run_model 3 "LLaMA-3-8B-Instruct" &
run_model 1 "GPT-2-XL" &
wait

python scripts/aggregate_results.py --track track1_general

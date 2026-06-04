#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mmlu_pro}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export HF_HOME="${HF_HOME:-$PWD/data/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/data/hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/data/hf_cache/datasets}"

run_one() {
  local gpu="$1"
  local model="$2"
  local dataset="$3"
  local safe_model="${model// /_}"
  local log_file="logs/track1_general/small/${dataset}/${safe_model}.log"
  mkdir -p "$(dirname "$log_file")"
  echo "[$(date)] starting ${model} on ${dataset} using physical GPU ${gpu}" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_track.py \
    --track track1_general \
    --model "$model" \
    --dataset "$dataset" \
    --overwrite >> "$log_file" 2>&1
  echo "[$(date)] finished ${model} on ${dataset}" | tee -a "$log_file"
}

python scripts/prepare_datasets.py --track track1_general
CUDA_VISIBLE_DEVICES=3 python scripts/check_track1_setup.py

mkdir -p "logs/track1_general/small/${DATASET}"
echo "[$(date)] ===== Track 1 small dataset start: ${DATASET} =====" | tee -a "logs/track1_general/small/master.log"

# Avoid physical GPU 0 because it is occupied by a VLLM process.
run_one 3 "LLaDA-8B" "$DATASET" &
run_one 1 "LLaDA-1.5" "$DATASET" &
run_one 4 "DREAM-7B" "$DATASET" &
run_one 2 "DiffuLLaMA-7B" "$DATASET" &
wait

run_one 3 "LLaMA-3-8B-Instruct" "$DATASET" &
run_one 1 "GPT-2-XL" "$DATASET" &
wait

python scripts/aggregate_results.py --track track1_general
echo "[$(date)] ===== Track 1 small dataset done: ${DATASET} =====" | tee -a "logs/track1_general/small/master.log"

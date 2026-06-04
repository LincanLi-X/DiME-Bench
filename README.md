# DiME-Bench

DiME-Bench is a mechanism-aware benchmark for discrete diffusion large language models (dLLMs). It evaluates when dLLMs benefit from bidirectional denoising, localized revision, constrained generation, and parallel decoding, rather than only reporting standard autoregressive-style leaderboard scores.

This repository contains the benchmark code, model/track configuration files, prompt templates, runners, parsers, and metrics. It does not include model checkpoints, HuggingFace dataset caches, generated outputs, logs, trajectories, manuscript files, or provisional simulation results.

## Repository Layout

```text
DiME-Bench/
├── requirements.txt
├── configs/
│   ├── global.yaml
│   ├── hardware.yaml
│   ├── models/
│   │   ├── ar/
│   │   └── dllm/
│   └── tracks/
├── prompts/
│   ├── track2_infilling/
│   ├── track3_editing/
│   ├── track4_reasoning/
│   ├── track5_constraints/
│   └── track6_efficiency/
├── scripts/
│   ├── prepare_*.py
│   ├── check_*.py
│   ├── run_*.py
│   └── aggregate_results.py
└── src/
    ├── data/
    ├── metrics/
    ├── models/
    ├── parsing/
    ├── tracks/
    └── utils/
```

## Tracks

DiME-Bench contains six evaluation tracks:

1. **Track 1: General Capability**  
   Sanity-check evaluation on knowledge, math, code, commonsense, and instruction following.

2. **Track 2: Text Infilling**  
   Prefix-suffix middle generation, testing bidirectional conditioning.

3. **Track 3: Text Editing and Revision**  
   Localized editing under minimal-change constraints, using edit success, preservation, and over-edit diagnostics.

4. **Track 4: Reasoning Ability**  
   Separates chaining-style reasoning from planning-style or global-consistency tasks.

5. **Track 5: Generation Quality and Constraints**  
   Open-ended response quality, instruction constraints, JSON/schema generation, and SQL execution.

6. **Track 6: Efficiency and Decoding Behavior**  
   Denoising-step, output-length, and batch-size sweeps with trajectory diagnostics such as STR, TRC, and FOE when available.

## Installation

Use Python 3.10+ with CUDA-enabled PyTorch for real model runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some tasks require optional packages or system tools:

- HumanEval execution uses local Python execution with timeouts.
- BERTScore is optional for Track 2 and is only computed when the package is available.
- SQL execution in Track 5 requires SQLite databases for Spider-style samples.
- Full model runs require local GPU access.

## Model Checkpoints

Model configs live under `configs/models/` and track-specific inline configs live in `configs/tracks/*.yaml`. By default, checkpoints are expected under:

```text
models/checkpoints/dllm/<model-name>/
models/checkpoints/ar/<model-name>/
```

Download configured Track 1 models with:

```bash
python scripts/download_models.py --track track1_general --model all
```

For gated models, authenticate with HuggingFace first or pass a token through the script option. Large model checkpoints are intentionally not included in this repository.

## Data Preparation

Processed datasets are written under `data/processed/<track>/<dataset>/`. Dataset caches are not included in this code package.

Prepare Track 1:

```bash
python scripts/prepare_datasets.py --track track1_general --dataset all
```

Prepare other tracks:

```bash
python scripts/prepare_track2_infilling_datasets.py --dataset all
python scripts/prepare_track3_editing_datasets.py --dataset all
python scripts/prepare_track4_reasoning_datasets.py --dataset all
python scripts/prepare_track5_constraints_datasets.py --dataset all
python scripts/prepare_track6_efficiency_datasets.py --dataset all
```

Some tracks can use small synthetic or fixture fallbacks for smoke tests. These are useful for checking code paths, but should not be used as final benchmark results unless explicitly intended.

## Setup Checks

Run setup checks before launching long jobs:

```bash
python scripts/check_track1_setup.py
python scripts/check_track2_infilling_setup.py --allow-missing-models
python scripts/check_track3_editing_setup.py --allow-missing
python scripts/check_track4_reasoning_setup.py --allow-missing-models
python scripts/check_track5_constraints_setup.py --allow-missing-models
python scripts/check_track6_efficiency_setup.py --allow-missing-models --allow-missing-upstream
```

Use the `--allow-*` flags for development machines where not all checkpoints or upstream data are available yet.

## Running Evaluations

Track 1:

```bash
python scripts/run_track.py --track track1_general --model LLaDA-8B --dataset gsm8k
python scripts/aggregate_results.py --track track1_general
```

Track-specific runners:

```bash
python scripts/run_track2_infilling.py --model LLaDA-8B --dataset wikitext103_sentence
python scripts/run_track3_editing.py --model LLaDA-8B --dataset synthetic_contradiction_repair
python scripts/run_track4_reasoning.py --model LLaDA-8B --dataset gsm8k
python scripts/run_track5_constraints.py --model LLaDA-1.5 --dataset json_schema
python scripts/run_track6_efficiency.py --model LLaDA-8B --dataset fixed_length --sweep t
```

Small-run helper scripts are provided for quick development checks:

```bash
bash scripts/run_track1_small_dataset.sh
bash scripts/run_track2_infilling_small.sh
bash scripts/run_track3_editing_small.sh
bash scripts/run_track4_reasoning_small.sh
bash scripts/run_track5_constraints_small.sh
bash scripts/run_track6_efficiency_small.sh
```

## Outputs

Runners write artifacts to the following directories:

```text
outputs/<track>/<model>/<dataset>.jsonl
logs/<track>/<model>/<dataset>.jsonl
metrics/<track>/<model>/<dataset>.json
trajectories/<track>/<model>/<dataset>.jsonl
```

These output directories are generated at runtime and are not included in this code-only package.

Each output row generally includes:

- `sample_id`
- `prompt`
- `model_output`
- `parsed_output`
- `reference`
- `decoding_config`
- `score`
- `error`

## Custom Diagnostics

DiME-Bench pairs final-answer metrics with mechanism-aware diagnostics:

- **Preservation Score:** how much non-target source content is retained during editing.
- **Over-Edit Rate:** how often the model changes more than the minimum required edit budget.
- **Stable Token Ratio (STR):** whether newly finalized dLLM tokens already match the final output.
- **Token Revision Count (TRC):** how many distinct non-mask values a position takes during denoising.
- **Finalization-Order Entropy (FOE):** whether finalization is spatially dispersed or AR-like.

Trajectory metrics require per-step token states and mask-status logs. If a model wrapper cannot expose these, DiME-Bench still reports final task and system metrics while marking trajectory diagnostics as unavailable.

## Reproducibility Notes

- Default seed: `42`.
- Default dLLM denoising budget: `T=64`, except Track 6 sweeps.
- Default decoding is deterministic: temperature `0.0`, top-p `1.0`, and no sampling.
- AR baselines use greedy decoding with dataset-specific output budgets.
- Track configs fix sample sizes, prompt templates, max token budgets, and reporting paths.

## What Is Not Included

This code package intentionally excludes:

- model checkpoints under `models/checkpoints/`
- HuggingFace caches under `data/hf_cache/`
- processed datasets and benchmark outputs
- logs, metrics, trajectories, reports, and figures
- manuscript `.tex` files and bibliography assets
- provisional internal simulation tables

Regenerate these artifacts from the scripts above when needed.


# DiME-Bench

**DiME-Bench: Mechanism-Oriented Suite for Assessing Inference and
Capabilities of Diffusion Language Models**

DiME-Bench is a mechanism-aware evaluation framework for discrete diffusion
language models (dLLMs). It measures final-task quality and the behaviors that
distinguish iterative denoising from left-to-right autoregressive (AR)
generation: bidirectional conditioning, localized revision, and
planning-oriented reasoning.

The repository is an installable Python package with frozen data contracts,
versioned prompts and parsers, AR/dLLM adapters, mechanism diagnostics,
resumable artifacts, and single- or multi-GPU execution backends.

> **Status:** research preview. The v1 benchmark protocol is frozen, while the
> reporting, unified CLI, CI, and public leaderboard layers remain under active
> development.

## Highlights

- **Four mechanism-aligned tracks.** General capability is a control; text
  infilling, localized editing, and reasoning structure probe dLLM-specific
  behavior.
- **Controlled AR/dLLM comparison.** Models share processed samples, semantic
  instructions, answer contracts, output budgets, deterministic settings, and
  failure policy.
- **Mechanism diagnostics.** Boundary consistency, contradiction, edit
  success, preservation, over-editing, and reasoning-regime aggregates augment
  standard task scores.
- **Auditable artifacts.** Every sample score links to an immutable raw
  prediction, resolved configuration, prompt hash, dataset hash, model
  revision, and environment snapshot.
- **Native dLLM inference.** The LLaDA adapter supports fixed-canvas masked
  denoising, bounded-middle infilling, configurable NFEs, block length,
  schedule, and unmasking policy.
- **Resumable execution.** Prediction-level caching, append-only traces,
  duplicate protection, worker retries, and isolated model × task directories
  make interrupted experiments recoverable.
- **Portable scheduling.** Run sequentially on one GPU, distribute independent
  jobs across a single multi-GPU node, or generate isolated Slurm scripts.

## Benchmark tracks

| Track | Evaluation role | Representative datasets | Main metrics |
|---|---|---|---|
| **1. General Capability** | Calibrate overall model strength | MMLU-Pro, HellaSwag, GSM8K, HumanEval, IFEval | Accuracy, normalized exact match, pass@1, instruction following |
| **2. Text Infilling** | Test bidirectional prefix/suffix conditioning | WikiText-103, CNN/DailyMail, arXiv abstracts | Token F1, ROUGE-L, BERTScore, boundary consistency, contradiction rate |
| **3. Text Editing** | Test targeted correction and edit locality | CoNLL-2014, FRUIT, Synthetic Repair | Edit success, preservation, over-edit rate, contradiction reduction |
| **4. Reasoning** | Contrast chain-style and planning-style structure | GSM8K, MATH-500, WinoGrande, Path-Star | Exact match, option accuracy, path validity, chain/planning averages |

See the [human-readable benchmark specification](docs/specification/benchmark-v1.md)
and the [machine-readable v1 protocol](specification/benchmark-v1.json) for the
authoritative task-to-metric mapping.

## Installation

DiME-Bench supports Python 3.10-3.12.

### Development installation

```bash
git clone https://github.com/DiME-Bench/DiME-Bench.git
cd DiME-Bench

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Optional dependencies

Install only the backends required by the experiment:

```bash
# Hugging Face datasets and autoregressive models
python -m pip install -e ".[hf]"

# Masked-diffusion model execution
python -m pip install -e ".[diffusion]"

# BERTScore, ROUGE, NumPy/SciPy, and related metrics
python -m pip install -e ".[metrics]"

# Code-execution evaluation support
python -m pip install -e ".[code]"

# Typical full research environment
python -m pip install -e ".[dev,hf,diffusion,metrics,code]"
```

Verify the installation:

```bash
dimebench --version
dimebench --help
```

## Quick start

### 1. Prepare deterministic sample data

The bundled sample mode is small, offline, and intended for schema and pipeline
validation:

```bash
dimebench prepare --suite dime_bench_v1
dimebench data validate data/processed/dime_bench_v1
```

Each processed split receives content-addressed sample IDs, an ordered dataset
hash, and a split lock. Repeating preparation with the same manifest, seed, and
selection parameters produces the same sample set.

Use `--full` for upstream data:

```bash
dimebench prepare \
  --suite minimal_4track \
  --full \
  --sample-limit 25 \
  --seed 0
```

External datasets are downloaded through their declared source adapters.
Synthetic Repair is generated locally by a frozen deterministic builder; it
does not require an external download. Datasets with redistribution or access
restrictions remain manual.

### 2. Inspect prompts without loading a model

```bash
dimebench task preview \
  --task wikitext103 \
  --samples data/processed/dime_bench_v1/wikitext103/test.jsonl \
  --model-family both \
  --limit 1
```

AR and diffusion templates share a semantic contract while retaining the
format required by each model family. Rendered prompt, template, and semantic
hashes are stored with every request.

### 3. Run the CPU-only mock pipeline

```bash
dimebench config validate configs/smoke.yaml
dimebench infer --config configs/smoke.yaml
```

Run the same command again to verify prediction-level resume and cache reuse.
The mock adapter allocates no model weights and is suitable for local
development and CI.

## Real-model evaluation

Tracked model configurations currently include:

- `meta-llama/Meta-Llama-3-8B-Instruct` through the Hugging Face causal-LM
  adapter;
- `GSAI-ML/LLaDA-8B-Instruct` through the native masked-diffusion adapter.

The tracked minimal evaluation matrix runs one selected dataset in every track
with both models:

| Track | Dataset | Samples | dLLM NFEs |
|---|---|---:|---:|
| General capability | GSM8K | 25 | 16 |
| Text infilling | WikiText-103 | 25 | 16 |
| Text editing | Synthetic Repair | 25 | 16 |
| Reasoning | WinoGrande | 25 | 16 |

The matrix is stored in `configs/experiments/minimal_4track.yaml`. It produces
eight isolated run directories and independently regenerates every score from
saved raw predictions.

## Configuration

DiME-Bench uses strict YAML configurations. Unknown fields and inconsistent
cross-references fail before model execution.

```text
configs/
├── models/
│   ├── autoregressive/
│   └── diffusion/
├── tasks/
│   ├── track1_general/
│   ├── track2_infilling/
│   ├── track3_editing/
│   └── track4_reasoning/
├── experiments/
└── runtime/
```

Typed dotted-key overrides are available for complete run configurations:

```bash
dimebench config validate configs/smoke.yaml --set batch_size=4
dimebench infer \
  --config configs/smoke.yaml \
  --set output_dir=/path/to/results
```

The stable configuration hash is computed after all overrides are applied.

## Model adapters and controlled decoding

All model backends implement one lifecycle-safe interface:

```text
load -> generate / score_options / denoise -> close
```

The common request schema separates prompts from model implementation. AR runs
use deterministic greedy decoding in the main protocol. Diffusion runs declare
their denoising steps, schedule, unmasking strategy, mask policy, generation
canvas, and block length. The benchmark records forward passes, token counts,
latency, and peak memory instead of treating an AR token step as equivalent to
a diffusion NFE.

See the [controlled-comparison protocol](docs/specification/model-protocol.md).

## Evaluation and mechanism diagnostics

Raw model text is retained unchanged. A separate postprocessing stage applies
versioned parsers and conservative failure detection before metrics run.

Standard metrics include accuracy, normalized exact match, pass@1, IFEval,
token F1, ROUGE-L, BERTScore F1, option accuracy, and path validity.

Mechanism diagnostics include:

- **Track 2:** left/right boundary consistency, contradiction rate, span-length
  and boundary-density stratification;
- **Track 3:** edit success, preservation score, over-edit rate, and
  contradiction reduction;
- **Track 4:** chain average, planning average, and reasoning-regime gap.

Neural NLI diagnostics use revision-pinned Transformers checkpoints. Missing
semantic inputs produce an explicit `null` with coverage metadata rather than
silently removing the sample. Failed primary-task predictions remain in the
denominator with a score of zero. Exact definitions are in
[metric-definitions.md](docs/specification/metric-definitions.md).

## Run artifacts

A fully evaluated model × task run owns a private directory:

```text
outputs/<run_id>/
├── run_manifest.json
├── environment.json
├── predictions.jsonl
├── inference-trace.jsonl
├── postprocessed-results.jsonl
├── sample_metrics.jsonl
├── summary.json
├── evaluation-report.json
└── recomputed/
    ├── postprocessed-results.jsonl
    ├── sample_metrics.jsonl
    └── summary.json
```

The manifest stores the resolved configuration, model/tokenizer revisions,
adapter version, dataset and ordered-sample hashes, environment, and hashes of
the canonical artifacts. `sample_metrics.jsonl` links each score to the exact
prediction hash. Final summaries retain contributing sample IDs and coverage.

## Single- and multi-GPU runners

DiME-Bench provides three execution backends:

- `LocalRunner`: sequential jobs on one GPU;
- `DistributedRunner`: deterministic, cost-aware data parallelism across GPUs
  on one node;
- `SlurmRunner`: isolated sbatch script generation with explicit submission.

Each `BenchmarkJob` represents one unique model × task pair and must own a
unique output directory. Workers receive an isolated `CUDA_VISIBLE_DEVICES`,
write per-attempt stdout/stderr logs, and retry without deleting prior
predictions or checkpoints.

Runtime profiles are tracked under `configs/runtime/`:

```text
local_1gpu.yaml
rtx6000_2gpu.yaml
b200_4gpu.yaml
```

Minimal Python usage:

```python
from dimebench.runners import DistributedRunner, load_runtime_config

runtime = load_runtime_config("configs/runtime/rtx6000_2gpu.yaml")
runner = DistributedRunner(runtime, log_dir="outputs/worker-logs")
report = runner.run(jobs)  # tuple[BenchmarkJob, ...]
```

This is one-process-per-GPU data parallelism across independent benchmark jobs,
not tensor/model parallelism within one model. Runner contract tests verify
that sequential and distributed execution produce identical scores and that
retries preserve durable checkpoints.

## Validation

Validate the frozen protocol and bundled data:

```bash
python scripts/validate_specification.py
python scripts/prepare_data.py --suite dime_bench_v1
python scripts/validate_data.py data/processed/dime_bench_v1
```

Full local quality checks:

```bash
ruff check dimebench scripts tests
mypy dimebench scripts
pytest -q
python -m build
twine check dist/*
```

Real-model evaluation requires a CUDA-enabled environment and appropriately
licensed model checkpoints. It is deliberately separate from the default
CPU-only test suite.

## Repository layout

```text
dimebench/
├── configs/               # model, task, experiment, and runtime configs
├── data/                  # manifests, cards, registry, and small fixtures
├── dimebench/
│   ├── artifacts/         # manifests, prediction/metric stores, provenance
│   ├── config/            # strict YAML loading and overrides
│   ├── datasets/          # acquisition, normalization, split locking
│   ├── decoding/          # shared budgets and diffusion sampling
│   ├── evaluators/        # standard and mechanism-aware metrics
│   ├── inference/         # batching, cache, retry, resume, traces
│   ├── models/            # mock, Hugging Face AR, and LLaDA adapters
│   ├── postprocessors/    # parsers and failure classification
│   ├── prompts/           # versioned AR/dLLM prompt contracts
│   ├── runners/           # local, multi-GPU, resource, and Slurm backends
│   ├── schemas/           # strict public data contracts
│   ├── tasks/             # track-specific task implementations
│   └── tracks/            # end-to-end track orchestration
├── docs/                  # protocol and validation guides
├── scripts/               # reproducible validation entry points
├── specification/         # machine-readable benchmark v1
└── tests/                 # unit, contract, integration, and GPU gates
```

The outer `dimebench/` is the repository root. The inner `dimebench/` is the
importable Python package, so users write `import dimebench` after installation.

## Extending DiME-Bench

- **Model:** implement `ModelAdapter`, declare capabilities, register it, and
  add contract tests plus a revision-pinned model config.
- **Dataset:** add a source manifest, deterministic preprocessor, license/data
  card, split lock, and registry entry.
- **Task:** bind a normalized dataset to a versioned prompt, parser, metrics,
  and output-token limit.
- **Metric:** implement a sample-level evaluator with a complete configuration
  hash and explicit eligibility behavior.
- **Runtime:** construct isolated `BenchmarkJob` objects or add a scheduler
  backend without changing inference or scoring semantics.

Changes to sample selection, prompt semantics, parser behavior, metric formulas,
aggregation, or failure treatment require a benchmark protocol version bump.

## Reproducibility policy

Headline results must retain:

- model and tokenizer revisions;
- processed dataset manifest and ordered sample hashes;
- prompt, parser, evaluator, and adapter versions;
- complete AR/dLLM decoding controls;
- raw predictions, per-sample metrics, and aggregate contributors;
- software environment and hardware metadata;
- explicit failures, null diagnostics, and coverage.

See [failure-policy.md](docs/specification/failure-policy.md) for retry,
failure-retention, and metric-eligibility rules.

## Citation

The paper citation will be added when the DiME-Bench manuscript is publicly
released. Until then, please cite the repository URL and the protocol version
used by your experiment.

## Acknowledgements

The repository organization and configuration-first user experience are
inspired in part by [OpenCompass](https://github.com/open-compass/opencompass).
DiME-Bench implements a distinct mechanism-oriented protocol, dLLM decoding
controls, diagnostics, and AR/dLLM comparison workflow.

## License

DiME-Bench is released under the [Apache License 2.0](LICENSE). Individual
datasets and model checkpoints remain subject to their original licenses and
access conditions.

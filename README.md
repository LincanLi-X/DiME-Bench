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

### Development installation❗️❗️ 

> The actual link of DiME-Bench is not given. Below the `User_Name` is shown as "DiME-Bench" to follow the double-blind review rule.
```bash
git clone https://github.com/DiME-Bench/DiME-Bench.git
cd DiME-Bench

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Installation during anonymous review

> [!IMPORTANT]
> This section applies during the double-blind review period. The source code
> is distributed exclusively through an anonymized, read-only repository.
> Please use the anonymous link below and select **Download ZIP**.
> 1. Open the [anonymous DiME-Bench repository](https://anonymous.4open.science/r/DiME-Bench-A6F4/).
> 2. 2. Select **Download ZIP** and extract the downloaded archive.
>    3. 3. Open a terminal in the extracted repository and run:

```bash
cd /path/to/extracted/DiME-Bench

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

#### To validate the built release artifact rather than an editable checkout:

```bash
python -m build
python -m pip install dist/dime_bench-1.0.0-py3-none-any.whl
dimebench smoke --output-dir outputs
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

Inspect the registered assets:

```bash
dimebench list models
dimebench list tasks
```

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

Inspect the exact AR and diffusion prompts for a prepared sample:

```bash
dimebench inspect prompt \
  --task mmlu_pro \
  --sample-id mmlu_pro.test.07a93fcb35bb959d
```

Use `--full` for upstream data:

```bash
dimebench prepare \
  --suite step12_minimal \
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

### 3. Run the CPU-only end-to-end smoke pipeline

```bash
dimebench config validate configs/smoke.yaml
dimebench run --config configs/smoke.yaml
dimebench leaderboard --run-dir outputs/smoke_diffusion_infilling
```

The `run` command executes inference, parsing, per-sample evaluation, run
sealing, summary export, and traceability checks. Run it again to verify
prediction-level resume and cache reuse. The mock adapter allocates no model
weights and is suitable for local development and CI.

For an installed wheel, the shortest equivalent command is:

```bash
dimebench smoke --output-dir outputs
```

The same stages can also be invoked independently:

```bash
dimebench infer --config configs/smoke.yaml
dimebench evaluate --run-dir outputs/smoke_diffusion_infilling
dimebench summarize --run-dir outputs/smoke_diffusion_infilling
```

## Real-model evaluation

Tracked model configurations currently include:

- `meta-llama/Meta-Llama-3-8B-Instruct` through the Hugging Face causal-LM
  adapter;
- `GSAI-ML/LLaDA-8B-Instruct` through the native masked-diffusion adapter.

The Step 12 minimal acceptance matrix runs one selected dataset in every track
with both models:

| Track | Dataset | Samples | dLLM NFEs |
|---|---|---:|---:|
| General capability | GSM8K | 25 | 16 |
| Text infilling | WikiText-103 | 25 | 16 |
| Text editing | Synthetic Repair | 25 | 16 |
| Reasoning | WinoGrande | 25 | 16 |

The gate produces eight run directories and independently regenerates every
score from saved raw predictions. The model IDs may be replaced by local
checkpoint paths without changing the evaluation contract. See the
[paper-reproduction guide](docs/reproduction/paper.md) for the portable
end-to-end workflow.

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

## Aggregation and paper reproduction

Step 14 aggregates results in the fixed order `sample -> dataset -> Track ->
benchmark`. It never embeds paper scores in source code. Each exported cell
retains its source run IDs, contributing-sample hashes, and source summary
hashes.

Validate and independently recompute one sealed run:

```bash
dimebench evaluate --run-dir outputs/<run_id>
```

Summarize a collection of completed run directories:

```bash
dimebench summarize \
  --run-dir outputs/aaai2027/paper_full/runs \
  --output-dir reports/aaai2027/paper_full
```

Regenerate the tracked paper deliverables:

```bash
dimebench reproduce paper \
  --config configs/experiments/aaai2027/paper_full.yaml
```

The output contains machine-readable benchmark summaries, tidy dataset/Track/
benchmark CSV files, matched AR/dLLM comparisons, figure-input JSON/CSV, paper
tables in JSON/CSV/Markdown/LaTeX, a traceability manifest, and hashes for all
generated artifacts. The full-paper config fails when a mapped paper cell is
missing. Lightweight private checks can opt into partial-table generation in a
separate local experiment configuration.

## Single- and multi-GPU runners

Step 13 adds three execution backends:

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
not tensor/model parallelism within one model. Reproducible deployments should
confirm that one- and multi-GPU runs produce identical scores and that retry
preserves existing artifacts.

## Documentation, leaderboard, and v1 release

Start with the [Quick Start](docs/getting-started/quickstart.md), then see the
[architecture](docs/concepts/architecture.md),
[result formats](docs/concepts/results.md),
[Track overview](docs/tracks/overview.md), and
[paper reproduction guide](docs/reproduction/paper.md). Runnable extension
examples for models, datasets, tasks, and metrics are under `examples/`.

Public ranking accepts only the versioned, hash-checked
`result-submission.json`; a bare `benchmark-summary.json` is rejected:

```bash
python scripts/release_benchmark.py \
  --summary reports/aaai2027/paper_full/benchmark-summary.json
python scripts/build_leaderboard.py
```

DiME-Bench provides versioned schemas and utilities for packaging benchmark
outputs into hash-validated result submissions and generating a static
leaderboard. Generated result bundles and leaderboard artifacts are stored
locally and are not committed to this repository. See the
[result-submission guide](docs/guides/submit-results.md) for details.

Build the CPU smoke container with:

```bash
docker build -f containers/Dockerfile -t dime-bench:1.0.0 .
docker run --rm -v "$PWD/container-results:/results" dime-bench:1.0.0
```

## Validation

Run the public CPU quality gates locally:

```bash
python scripts/validate_step0.py
ruff check dimebench scripts tests
mypy dimebench scripts
pytest -q
python -m build
twine check dist/*
```

GPU tests are marked `gpu` and remain separate from the default CPU suite:

```bash
pytest -m gpu tests/gpu -ra
```

The repository CI workflows run lint/type checks, the CPU test matrix, and the
installed-wheel smoke evaluation. Real-model experiments remain explicit
research runs because they require gated checkpoints and accelerator access.

## Repository layout

```text
dimebench/
├── configs/               # model, task, experiment, and runtime configs
├── containers/            # reproducible CPU smoke image recipe
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
│   ├── reporting/         # paper tables, LaTeX, and plotting inputs
│   ├── runners/           # local, multi-GPU, resource, and Slurm backends
│   ├── schemas/           # strict public data contracts
│   ├── summarizers/       # dataset, Track, benchmark, AR/dLLM aggregation
│   ├── tasks/             # track-specific task implementations
│   ├── tracks/            # end-to-end track orchestration
│   └── workflow.py        # unified infer -> evaluate -> seal workflow
├── docs/                  # user, extension, protocol, reproduction guides
├── examples/              # model, dataset, task, and metric extensions
├── leaderboard/           # static versioned leaderboard application
├── results/releases/      # immutable content-addressed releases
├── scripts/               # data, reproduction, release, leaderboard tools
├── specification/         # machine-readable benchmark v1
└── tests/                 # unit, contract, golden, integration, smoke, GPU
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


## Citation

```
@inproceedings{anonymous2026dimebench,
  title     = {{DiME-Bench}: Mechanism-Oriented Suite for Assessing Inference and Capabilities of Diffusion Language Models},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2026}
}
```


## License

DiME-Bench is released under the [Apache License 2.0](LICENSE). Individual
datasets and model checkpoints remain subject to their original licenses and
access conditions.

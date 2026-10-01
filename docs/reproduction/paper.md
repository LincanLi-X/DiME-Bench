# Reproduce the DiME-Bench paper outputs

## Inputs

Paper reproduction consumes completed run directories, not handwritten table
values. Each run must contain a completed manifest, predictions, sample
metrics, summary, and matching artifact hashes.

## Recompute and aggregate

```bash
python scripts/validate_results.py \
  --run-root outputs/aaai2027/paper_full/runs \
  --output reports/run-validation.json

dimebench reproduce paper \
  --config configs/experiments/aaai2027/paper_full.yaml
```

The reproduction command writes dataset, Track, benchmark, AR/dLLM comparison,
figure-input, Markdown, CSV, JSON, and LaTeX artifacts. Full-paper mode fails
when a mapped paper cell is missing.

## Build the v1 release

```bash
python scripts/release_benchmark.py \
  --summary reports/aaai2027/paper_full/benchmark-summary.json
python scripts/build_leaderboard.py
```

`results/releases/v1.0.0/release-manifest.json` records a SHA-256 for every
release file. The leaderboard copy of `benchmark-summary.json` must have the
same file hash as the release copy.

The committed reference release is generated from the eight-run Step 12
minimal validation matrix. It demonstrates traceability and publication flow;
the complete paper release should be regenerated from the full experiment
matrix using the same commands.

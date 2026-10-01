# Run, summary, and submission formats

## Run directory

Each model-task pair owns one directory. The canonical artifacts are:

| File | Purpose |
|---|---|
| `run_manifest.json` | Resolved configuration, provenance, sample set, status, hashes |
| `environment.json` | Python, package, Git, hardware, and dependency versions |
| `predictions.jsonl` | Immutable requests and raw model responses |
| `postprocessed-results.jsonl` | Parsed outputs and explicit failure labels |
| `sample_metrics.jsonl` | Per-sample values linked to prediction hashes |
| `summary.json` | Metric aggregates, contributors, coverage, and failures |
| `evaluation-report.json` | Independent recomputation outcome |

Files use strict schema version `1.0`. A failed or skipped prediction remains
in primary-metric denominators; an ineligible mechanism diagnostic is `null`
with explicit coverage metadata.

## Benchmark summary

`benchmark-summary.json` aggregates in the fixed order sample → dataset →
Track → benchmark. It contains the exact source run IDs and a
`source_fingerprint` derived from the contributing run configurations and
artifact hashes.

## Public result submission

The leaderboard does not accept a bare summary. It accepts
`result-submission.json`, validated against
`specification/result-submission-v1.schema.json`. The envelope fixes:

- benchmark ID and version;
- submission ID syntax;
- the complete typed `BenchmarkSummary`;
- a SHA-256 of the canonical embedded summary.

Changing any score, source run, or traceability field invalidates the
submission before ranking.

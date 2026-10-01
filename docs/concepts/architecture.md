# Architecture

DiME-Bench separates five concerns so that model execution is replaceable
without changing task semantics or scores:

```text
Dataset manifest -> normalized SampleRecord -> Task + versioned prompt
    -> ModelRequest -> ModelAdapter -> immutable raw prediction
    -> versioned parser -> ResultSpec -> evaluator -> sample metric
    -> dataset -> Track -> benchmark summary -> versioned submission
```

## Configuration layer

`RunSpec` binds one model, one normalized dataset split, one task, and one
decoding budget. Strict Pydantic schemas reject unknown keys. The resolved
configuration is hashed and stored in the run manifest.

## Data and task layers

The data layer owns download, normalization, stable sample IDs, split locks,
and hashes. The task layer owns semantic instructions, matched AR/dLLM prompt
templates, parser selection, metrics, and output budgets. Gold answers are
never exposed by the prompt renderer.

## Model and inference layers

Every backend implements the same lifecycle: `load`, inference, and `close`.
AR and diffusion adapters return a common `ModelResponse`. The inference engine
adds batching, caching, retries, traces, and prediction-level resume.

## Evaluation and publication layers

Evaluation reads saved predictions and writes per-sample metrics before
aggregation. Publication consumes a sealed `BenchmarkSummary`, embeds it in a
v1 result submission with a canonical SHA-256, and builds the leaderboard only
after schema and hash validation.

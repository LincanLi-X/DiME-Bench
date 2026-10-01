# Failure and Retention Policy

## Core rule

Every scheduled sample remains in the run record. Failures are labeled and
counted; they are never silently removed to improve scores.

## Failure classes

| ID | Definition | Primary task treatment |
|---|---|---|
| `empty_output` | Empty or whitespace-only model output | Incorrect / zero |
| `parse_error` | Non-empty output cannot be parsed into required format | Incorrect / zero |
| `context_copy` | Infilling output copies supplied context instead of producing only the missing span | Incorrect / zero |
| `full_document_output` | Infilling model returns prefix and/or suffix with the middle span | Incorrect / zero |
| `truncated_output` | Generation ends at the declared limit before a valid answer/end marker | Incorrect / zero unless official task rule says otherwise |
| `runtime_error` | Model, API, code execution, or evaluator fails after configured retries | Incorrect / zero for task success; error retained |
| `unsafe_code_result` | Code evaluation violates sandbox policy or times out | Failed test / zero |

## Detection rules

- Detection runs after conservative text normalization and before task scoring.
- Original output is never overwritten.
- Each detector has a version and stores evidence, such as matched context span,
  parser exception, finish reason, timeout, or traceback digest.
- A sample may have multiple failure labels; `runtime_error` takes precedence
  when no usable model output exists.

## Metric eligibility

- Primary accuracy, exact-match, pass-rate, and generation-quality metrics
  assign the task-defined failure value, normally zero.
- A mechanism metric is computed only when its required semantic inputs exist.
- Ineligible mechanism values are `null`, never silently imputed.
- Every mechanism aggregate reports eligible count, total scheduled count,
  coverage, and failure counts so models cannot benefit from missing outputs.
- Leaderboards must display coverage and format-failure rate beside mechanism
  metrics.

## Retry policy

- Deterministic local model failures may be retried once after clearing the
  failed batch.
- Transient API failures use the configured bounded retry policy.
- Retried samples preserve attempt metadata; only the declared final attempt is
  scored.
- Changing decoding parameters during a retry creates a new run, not a retry.

## Exclusion policy

Samples may be excluded only before model execution for a documented dataset
integrity reason, such as corrupted source data or license restriction. Such
exclusions require a new frozen manifest and cannot be decided from model
outputs.


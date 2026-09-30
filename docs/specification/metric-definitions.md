# Metric Definitions

## General rules

- Metrics operate on retained samples; failed samples are never silently
  deleted.
- Each metric result records metric ID, version, value, denominator, coverage,
  and evaluator configuration hash.
- Percent-valued paper tables may display `[0, 1]` scores multiplied by 100.
- Dataset metrics are macro-averaged over eligible samples unless an official
  evaluator defines otherwise.

## Standard metrics

### `accuracy`

Mean correctness of parsed options. Values: `[0, 1]`; higher is better.

### `normalized_exact_match`

Mean exact match after the task's versioned normalization, including numeric
canonicalization for GSM8K/MATH-500. Values: `[0, 1]`; higher is better.

### `pass_at_1`

Fraction of one greedy code completion per problem that passes all official
unit tests. Values: `[0, 1]`; higher is better.

### `instruction_following_rate`

Official IFEval rule-based pass rate. The selected official aggregation and
checker revision must be recorded. Values: `[0, 1]`; higher is better.

### `token_f1`

Token-overlap F1 between predicted and reference middle spans after versioned
normalization. Values: `[0, 1]`; higher is better.

### `rouge_l`

ROUGE-L F1 between predicted and reference middle spans. Values: `[0, 1]`;
higher is better.

### `bertscore_f1`

BERTScore F1 using a frozen scorer model, revision, tokenizer, IDF policy, and
rescaling setting. Values: scorer-defined normalized range; higher is better.

### `option_accuracy`

Mean correctness of the parsed WinoGrande option. Values: `[0, 1]`; higher is
better.

### `path_validity`

Fraction of Path-Star outputs that parse into a path satisfying the official
start, goal, adjacency, and obstacle constraints. Values: `[0, 1]`; higher is
better.

## Track 2 mechanism metrics

### `boundary_consistency`

For each valid generated middle span, a versioned deterministic boundary
evaluator scores the left transition `(prefix, middle)` and right transition
`(middle, suffix)` as consistent (`1`) or inconsistent (`0`). The sample score
is the mean of both boundary scores; the dataset score is their macro mean.
The evaluator prompt/model or rule implementation, context-window construction,
revision, and threshold form part of the metric configuration hash. Values:
`[0, 1]`; higher is better.

The v1 implementation evaluates `(prefix, generated_middle)` and
`(generated_middle, suffix)` separately, thresholds each frozen pair-classifier
probability into `0/1`, and averages the two binary outcomes. Both raw
probabilities are retained in sample metadata.

### `contradiction_rate`

Fraction of eligible samples for which a versioned contradiction evaluator
detects at least one semantic contradiction between the generated middle and
either supplied boundary. The evaluator backend, revision, sentence splitting,
threshold, and aggregation rule must be frozen in configuration. Values:
`[0, 1]`; lower is better.

The v1 implementation applies the frozen NLI backend to the generated middle
against each supplied boundary and returns `1` when either contradiction
probability reaches the configured threshold. Backend model ID, immutable
revision, maximum input length, device class, threshold, and `any_boundary`
aggregation are hashed.

### Span and boundary strata

Span-length bins and low/high boundary-density labels are stored in the frozen
dataset manifest. They are grouping variables, not model-generated labels.

## Track 3 mechanism metrics

All edit-distance and LCS operations use the benchmark's versioned tokenizer
and normalization pipeline.

### `edit_success`

Dataset-specific success from an official checker, reference comparison,
rule-based validator, or frozen entailment evaluator. The route is fixed per
dataset and recorded. Values: `[0, 1]`; higher is better.

### `preservation_score`

Let `N_src` be normalized source tokens outside the target region and `N_out`
the aligned output tokens corresponding to non-target content:

```text
Preservation = LCS(N_src, N_out) / max(len(N_src), 1)
```

Values: `[0, 1]`; higher is better.

`N_src` comes from declared non-target spans when available. Otherwise it is
the source-token subsequence aligned to equal regions of the source/reference
diff. `N_out` is the normalized full output; target tokens cannot increase the
LCS unless they also match frozen non-target content. Samples with no
non-target tokens are ineligible rather than assigned an artificial score.

### `over_edit_rate`

Let `d_actual` be token edit distance from source to output and `d_min` the
minimum required edit distance stored in the sample:

```text
OverEdit = max(d_actual - d_min, 0) / max(d_actual, 1)
```

If `d_actual=0`, the score is `0`; edit success separately captures failure to
make a required change. Values: `[0, 1]`; lower is better. A non-target edit
ratio may be reported as an additional diagnostic but does not replace this
primary definition.

The v1 tokenizer is NFKC-normalized, case-folded alphanumeric tokenization and
`d_actual` is unit-cost token Levenshtein distance.

### `contradiction_reduction_rate`

Among samples containing a source contradiction, this is the fraction for
which the output removes the target contradiction without introducing a new
contradiction under the frozen evaluator. Values: `[0, 1]`; higher is better.

Eligibility requires supplied evidence and a source contradiction detected at
the frozen threshold. The score is `1` only when no supplied evidence remains
contradictory with the edited output. Missing evidence or an undetected source
contradiction produces `null` with an explicit ineligibility reason.

## Track 4 aggregates

### `chain_average`

Macro mean of normalized GSM8K and MATH-500 scores. It is `null` unless both
task results exist for the model.

### `planning_average`

Macro mean of normalized WinoGrande and Path-Star scores. It is `null` unless
both task results exist for the model.

### `reasoning_regime_gap`

```text
ReasoningRegimeGap = PlanningAverage - ChainAverage
```

Positive values indicate higher planning-style performance relative to
chain-style performance. This is diagnostic, not an overall capability score.

## Evaluator implementation freeze

The v1 protocol freezes metric contracts. Neural judge/NLI implementations are
revision-pinned. A result is reproducible only when its evaluator ID, model revision,
prompt hash, threshold, and configuration hash are recorded; results from
different evaluator hashes must not be merged.

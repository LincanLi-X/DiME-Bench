# DiME-Bench v1 Benchmark Specification

## Status

- Protocol version: `1.0.0`
- Scope: text-only discrete diffusion language models and matched autoregressive baselines
- Tracks: four
- Machine-readable source: `specification/benchmark-v1.json`

This document freezes the public evaluation contract. Implementations may
optimize execution, but must not change task inputs, sample IDs, prompt
semantics, parsers, metric definitions, or aggregation rules without a new
protocol version.

## Design principles

1. **Mechanism alignment:** general capability is a control; infilling,
   editing, and reasoning structure expose mechanism-relevant behavior.
2. **Controlled comparison:** AR and dLLM systems use the same processed
   samples, semantic task instructions, output contracts, parser versions,
   hardware class, and declared output limits.
3. **Mechanism-aware metrics:** task success is reported together with
   boundary, contradiction, locality, and reasoning-regime diagnostics.
4. **Reproducibility:** every run records model/tokenizer revisions, prompt
   hash, dataset manifest hash, decoding configuration, software versions,
   hardware, and per-sample outputs.

## Track 1: General LLM Capability

**Role:** calibration control; it is not evidence of a diffusion-specific
mechanism advantage.

| Task | Capability | Primary metric | Main output contract |
|---|---|---|---|
| MMLU-Pro | Knowledge and multi-domain reasoning | Accuracy | One option |
| HellaSwag | Commonsense completion | Accuracy | One option |
| GSM8K | Grade-school mathematics | Normalized exact match | Final numeric answer |
| HumanEval | Code generation | pass@1 | Executable completion |
| IFEval | Instruction following | Official instruction-following rate | Instruction-compliant response |

Multiple-choice tasks use normalized option likelihood when the adapter
supports it; otherwise they use deterministic direct-answer generation. The
selected scoring route is recorded per run and must not change within a
model-task run.

## Track 2: Text Infilling and Middle Generation

**Role:** evaluate bidirectional prefix-suffix conditioning.

Each sample contains prefix `P`, gold middle span `M`, suffix `S`, domain,
span length, prefix/suffix lengths, and boundary-density label. The submitted
answer must contain only the missing middle span.

| Task | Setting | Task metric | Mechanism diagnostics |
|---|---|---|---|
| WikiText-103 | Sentence/clause infilling | Token F1 | Boundary Consistency, Contradiction Rate |
| CNN/DailyMail | Paragraph-level news infilling | ROUGE-L | Boundary Consistency, Contradiction Rate |
| arXiv abstracts | Scientific abstract infilling | BERTScore F1 | Boundary Consistency, Contradiction Rate |

Results are also stratified by short/medium/long span and low/high boundary
density. dLLMs receive a bounded masked region between `P` and `S`; AR models
receive a structurally equivalent instruction containing both boundaries.

## Track 3: Text Editing and Revision

**Role:** measure correction quality and edit locality.

Each sample contains source text, edit instruction, reference edit, target
span when available, non-target spans, minimum required edit distance, and
edit type.

| Task | Setting | Required metrics |
|---|---|---|
| CoNLL-2014 GEC | Grammatical correction | Edit Success, Preservation, Over-Edit |
| FRUIT | Evidence-grounded factual editing | Edit Success, Preservation, Over-Edit, Contradiction Reduction |
| Synthetic repair | Known-span contradiction repair | Edit Success, Preservation, Over-Edit, Contradiction Reduction |

The primary protocol is instruction-only full-text editing. A target-marked
synthetic diagnostic is a separate setting and must never be mixed with the
primary results.

## Track 4: Reasoning Ability

**Role:** compare sequential chaining with globally constrained planning.

| Reasoning type | Tasks | Metrics |
|---|---|---|
| Chain | GSM8K, MATH-500 | Normalized exact match, Chain Average |
| Planning | WinoGrande, Path-Star | Option accuracy, path validity, Planning Average |

The main setting uses deterministic direct-answer prompts. Few-shot or
chain-of-thought variants are separate experiments. Chain and Planning
averages are reported only when the model has results for both tasks in that
group.

## Paper-result mapping

- Table 2: five Track 1 task metrics.
- Table 3: three generation-quality metrics plus Boundary Consistency and
  Contradiction Rate.
- Table 4: Edit Success, Preservation, Over-Edit, and Contradiction Reduction.
- Table 5: four task scores plus Chain Average and Planning Average.

The complete column-to-task-to-metric mapping is validated from
`specification/benchmark-v1.json` by `scripts/validate_step0.py`.

## Versioning rule

Changes to a dataset split, prompt semantics, parser, metric formula,
aggregation rule, or failure treatment require a protocol version increment.
Formatting-only documentation changes do not.


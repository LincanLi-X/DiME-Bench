# Model and Controlled-Comparison Protocol

## Model families

- **dLLM:** token-discrete diffusion models that recover masked or corrupted
  token sequences through iterative denoising.
- **AR:** left-to-right autoregressive language models used as controlled
  baselines.

Continuous diffusion models and multimodal models are outside v1 scope.

## Shared comparison contract

Within each task, all compared models must share:

- processed split and ordered sample IDs;
- semantic task instruction and required answer format;
- parser and evaluator versions;
- maximum answer length declared by the task;
- deterministic main setting;
- hardware class where runtime is compared;
- failure-retention policy.

Model-specific chat wrappers are allowed only when required by the released
model. Their rendered prompts and hashes must be stored.

## dLLM decoding contract

- Default denoising steps: `T=128` unless a task config explicitly overrides it.
- Default dtype: `bfloat16` on supported GPU hardware.
- Default strategy: confidence-based unmasking.
- The run manifest must record mask length, step count, schedule, remasking
  policy, length-control rule, EOS/end-marker rule, seed, and batch size.
- Intermediate denoising trajectories are optional diagnostics and are not
  required for v1 headline metrics.

## AR decoding contract

- Main results use greedy decoding (`temperature=0`, no sampling).
- Each task declares a fixed maximum output-token budget.
- The run manifest records generation arguments, tokenizer revision, stopping
  rules, seed, and batch size.
- Infilling prompts must provide both prefix and suffix and request only the
  missing span.

## Budget interpretation

DiME-Bench matches task-visible output limits and deterministic conditions; it
does not claim that one AR token step equals one diffusion denoising step.
Compute and latency comparisons require separate reporting of forward passes,
denoising steps, generated tokens, wall-clock time, batch size, and hardware.

## Scoring routes

- Multiple choice: normalized option likelihood when supported; otherwise
  deterministic direct-answer parsing.
- Free generation: task-specific deterministic parser.
- Code: greedy completion followed by isolated unit-test execution.
- Instruction following: official rule-based checker.

The scoring route must be constant within a model-task run and recorded in the
manifest. Results using different routes must be visibly labeled.

## Required run metadata

```text
benchmark_version
model_id
model_family
model_revision
tokenizer_revision
adapter_version
dataset_manifest_hash
sample_ids_hash
prompt_id and prompt_hash
parser_id and parser_version
metric_ids and versions
decoding configuration
software environment
hardware description
start/end timestamps
```

## Paper model coverage

- Track 1 dLLMs: LLaDA-8B, LLaDA-1.5, DREAM-7B, DiffuLLaMA-7B.
- Track 2 dLLMs: Track 1 set plus DreamOn.
- Track 3 dLLMs: LLaDA-8B, LLaDA-1.5, DREAM-7B, TESS2, DreamOn, as assigned
  by dataset in the paper protocol.
- Track 4 dLLMs: LLaDA-8B, LLaDA-1.5, DREAM-7B, DreamOn, TESS2.
- Primary AR baseline: LLaMA-3-8B-Instruct.
- Complementary AR anchors: GPT-2-XL and gpt-oss-20b on their assigned tasks.

Adding a new model does not change benchmark v1, but its adapter, revision,
capabilities, and evaluation coverage must be declared.


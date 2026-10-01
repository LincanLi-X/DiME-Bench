# Benchmark tracks

## Track 1: General LLM capability

Track 1 is the calibration control. MMLU-Pro and HellaSwag use accuracy; GSM8K
uses normalized exact match; HumanEval uses pass@1; IFEval uses instruction
following. It distinguishes a general capability gap from a mechanism-specific
advantage or weakness.

## Track 2: Text infilling

Track 2 supplies both left and right boundaries and asks for only the missing
middle. It evaluates lexical/semantic similarity together with boundary
consistency, contradiction rate, span length, and boundary density. The task
directly probes bidirectional conditioning.

## Track 3: Text editing and revision

Track 3 asks the model to return a complete edited text. Edit success is
reported with preservation, over-edit rate, and contradiction reduction.
Target and non-target regions are retained when a dataset supplies them.

## Track 4: Reasoning structure

Track 4 separates chain-style arithmetic/deduction from planning-style option
and path tasks. DiME-Bench reports individual dataset metrics, chain and
planning averages, and the planning-minus-chain regime gap.

## Controlled comparison

AR and dLLM runs share normalized samples, semantic instructions, answer
contracts, maximum output lengths, deterministic settings, parsing, and
metrics. Diffusion-specific NFEs, schedule, unmasking, and mask policy remain
explicit rather than being treated as equivalent to AR token steps.

# Add a task and prompt

1. Create a strict `TaskSpec` selecting the dataset, Track, task type, parser,
   metrics, primary metric, and maximum output length.
2. Create one `PromptTemplateSpec` with a shared semantic contract and both AR
   and diffusion templates.
3. Implement `prompt_fields()` in a `BaseTask` subclass or use the matching
   built-in Track task.
4. Confirm that gold answers are absent from rendered fields.
5. Add contract tests for both model families and a prompt preview fixture.

Prompt semantic changes require a version bump because prompt hashes and run
configuration hashes are part of result provenance.

See [`examples/add_custom_task.py`](../../examples/add_custom_task.py).

"""Construct and render a custom task with paired AR/dLLM prompts."""

from __future__ import annotations

from dimebench.datasets import GenerationRecord
from dimebench.prompts import PromptSemanticContract, PromptTemplateSpec
from dimebench.schemas import TaskSpec
from dimebench.tasks import create_task


def main() -> None:
    spec = TaskSpec(
        id="example_qa",
        track="track1_general",
        dataset_id="example_qa",
        type="generation",
        prompt_id="example.qa",
        parser_id="raw_text",
        metrics=("normalized_exact_match",),
        primary_metric="normalized_exact_match",
        max_output_tokens=16,
    )
    prompt = PromptTemplateSpec(
        id="example.qa",
        version="1.0.0",
        track="track1_general",
        sample_kind="generation",
        semantic=PromptSemanticContract(
            instruction="Answer the question directly.",
            input_fields=("instruction",),
            output_contract="Return only the short answer.",
        ),
        templates={
            "autoregressive": "Question: {instruction}\nAnswer:",
            "diffusion": "Fill the answer for: {instruction}",
        },
    )
    sample = GenerationRecord(
        sample_id="example_qa.test.item-1",
        dataset_id="example_qa",
        split="test",
        source_id="item-1",
        instruction="What is 2 + 2?",
        references=("4",),
    )
    task = create_task(spec, prompt)
    for family in ("autoregressive", "diffusion"):
        rendered = task.render(sample, family)
        print(f"[{family}] {rendered.text}")


if __name__ == "__main__":
    main()

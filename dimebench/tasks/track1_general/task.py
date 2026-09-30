"""Prompt field extraction for general-capability tasks."""

from __future__ import annotations

from dimebench.datasets.records import (
    GenerationRecord,
    MultipleChoiceRecord,
    SampleRecord,
)
from dimebench.tasks.base import BaseTask, TaskError


def _choice_label(index: int) -> str:
    if index >= 26:
        raise TaskError("multiple-choice prompts support at most 26 options")
    return chr(ord("A") + index)


class GeneralTask(BaseTask):
    """Render Track 1 multiple-choice, generation, and code inputs."""

    def prompt_fields(self, sample: SampleRecord) -> dict[str, str]:
        if isinstance(sample, MultipleChoiceRecord):
            choices = "\n".join(
                f"{_choice_label(index)}. {choice}"
                for index, choice in enumerate(sample.choices)
            )
            return {
                "question": sample.question,
                "context": sample.context or "None provided.",
                "choices": choices,
            }
        if isinstance(sample, GenerationRecord):
            return {
                "instruction": sample.instruction,
                "context": sample.context or "None provided.",
            }
        raise TaskError(f"Track 1 cannot render sample kind {sample.kind!r}")


__all__ = ["GeneralTask"]

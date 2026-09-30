"""Prompt field extraction for prefix-suffix infilling."""

from __future__ import annotations

from dimebench.datasets.records import InfillingRecord, SampleRecord
from dimebench.tasks.base import BaseTask, TaskError


class InfillingTask(BaseTask):
    """Expose both boundaries while keeping the gold middle hidden."""

    def prompt_fields(self, sample: SampleRecord) -> dict[str, str]:
        if not isinstance(sample, InfillingRecord):
            raise TaskError(f"Track 2 cannot render sample kind {sample.kind!r}")
        return {"prefix": sample.prefix, "suffix": sample.suffix}


__all__ = ["InfillingTask"]

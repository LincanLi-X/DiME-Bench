"""Base task contract and content-addressed prompt rendering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import Field

from dimebench.artifacts.hashing import hash_json
from dimebench.datasets.records import SampleKind, SampleRecord
from dimebench.prompts import PromptTemplateSpec
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.model import ModelFamily
from dimebench.schemas.task import TaskSpec


class TaskError(ValueError):
    """Raised when a sample, task, and prompt contract are incompatible."""


class RenderedPrompt(StrictSchema):
    """Final prompt text plus immutable provenance required by inference."""

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    model_family: ModelFamily
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    semantic_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_contract: str = Field(min_length=1)
    text: str = Field(min_length=1)


_EXPECTED_SAMPLE_KIND: dict[str, tuple[SampleKind, ...]] = {
    "multiple_choice": ("multiple_choice",),
    "generation": ("generation",),
    "code": ("generation",),
    "infilling": ("infilling",),
    "editing": ("editing",),
    "reasoning": ("reasoning",),
}


class BaseTask(ABC):
    """Bind one TaskSpec to a versioned prompt pair."""

    def __init__(self, spec: TaskSpec, prompt: PromptTemplateSpec) -> None:
        if prompt.id != spec.prompt_id:
            raise TaskError(
                f"task {spec.id!r} expects prompt {spec.prompt_id!r}, got {prompt.id!r}"
            )
        if prompt.version != spec.prompt_version:
            raise TaskError(
                f"task {spec.id!r} expects prompt version "
                f"{spec.prompt_version!r}, got {prompt.version!r}"
            )
        if prompt.track != spec.track:
            raise TaskError(
                f"task {spec.id!r} track {spec.track!r} does not match "
                f"prompt track {prompt.track!r}"
            )
        expected_kinds = _EXPECTED_SAMPLE_KIND[spec.type]
        if prompt.sample_kind not in expected_kinds:
            raise TaskError(
                f"task type {spec.type!r} is incompatible with prompt sample kind "
                f"{prompt.sample_kind!r}"
            )
        self.spec = spec
        self.prompt = prompt

    @property
    def contract_hash(self) -> str:
        """Hash task scoring protocol and family-independent prompt semantics."""
        return hash_json(
            {
                "task": self.spec,
                "prompt_semantic_hash": self.prompt.semantic_hash,
            }
        )

    def render(self, sample: SampleRecord, family: ModelFamily) -> RenderedPrompt:
        """Render and hash the final prompt for one normalized sample."""
        if sample.dataset_id != self.spec.dataset_id:
            raise TaskError(
                f"task {self.spec.id!r} expects dataset {self.spec.dataset_id!r}, "
                f"got {sample.dataset_id!r}"
            )
        if sample.kind != self.prompt.sample_kind:
            raise TaskError(
                f"prompt {self.prompt.id!r} expects sample kind "
                f"{self.prompt.sample_kind!r}, got {sample.kind!r}"
            )
        fields = self.prompt_fields(sample)
        text = self.prompt.render(family, fields)
        template_hash = self.prompt.template_hash(family)
        prompt_hash = hash_json(
            {
                "sample_id": sample.sample_id,
                "task_id": self.spec.id,
                "model_family": family,
                "prompt_id": self.prompt.id,
                "prompt_version": self.prompt.version,
                "semantic_hash": self.prompt.semantic_hash,
                "template_hash": template_hash,
                "text": text,
            }
        )
        return RenderedPrompt(
            sample_id=sample.sample_id,
            dataset_id=sample.dataset_id,
            task_id=self.spec.id,
            model_family=family,
            prompt_id=self.prompt.id,
            prompt_version=self.prompt.version,
            semantic_hash=self.prompt.semantic_hash,
            template_hash=template_hash,
            prompt_hash=prompt_hash,
            output_contract=self.prompt.semantic.output_contract,
            text=text,
        )

    @abstractmethod
    def prompt_fields(self, sample: SampleRecord) -> dict[str, str]:
        """Expose only task inputs; gold answers must never be returned."""


__all__ = ["BaseTask", "RenderedPrompt", "TaskError"]

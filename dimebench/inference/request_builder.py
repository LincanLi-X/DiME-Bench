"""Build model-independent requests from normalized samples and Task prompts."""

from __future__ import annotations

from pathlib import Path

from dimebench.artifacts.hashing import hash_json
from dimebench.datasets.records import (
    InfillingRecord,
    MultipleChoiceRecord,
    SampleRecord,
)
from dimebench.models import Capability, ModelAdapter
from dimebench.prompts import PromptTemplateSpec, load_prompt_spec
from dimebench.schemas.request import ModelRequest, RequestMode
from dimebench.schemas.run import RunSpec
from dimebench.tasks import BaseTask, TaskError, create_task


class RequestBuilder:
    """Render prompts and select one adapter-compatible inference route."""

    VERSION = "1.0.0"

    def __init__(
        self,
        spec: RunSpec,
        adapter: ModelAdapter,
        prompt_root: str | Path,
    ) -> None:
        self.spec = spec
        self.adapter = adapter
        self.task = self._load_task(Path(prompt_root))

    def _load_task(self, prompt_root: Path) -> BaseTask:
        prompts: dict[str, PromptTemplateSpec] = {}
        for path in sorted(prompt_root.glob("track*/*.yaml")):
            prompt = load_prompt_spec(path)
            if prompt.id in prompts:
                raise TaskError(f"duplicate prompt id {prompt.id!r}")
            prompts[prompt.id] = prompt
        try:
            prompt = prompts[self.spec.task.prompt_id]
        except KeyError as exc:
            raise TaskError(
                f"unknown prompt {self.spec.task.prompt_id!r} under {prompt_root}"
            ) from exc
        return create_task(self.spec.task, prompt)

    def _mode_for(self, sample: SampleRecord) -> RequestMode:
        if isinstance(sample, MultipleChoiceRecord) and self.adapter.supports(
            Capability.SCORE_OPTIONS
        ):
            return "score_options"
        if self.spec.task.type == "infilling":
            return "infill"
        if self.spec.task.type == "editing":
            return "edit"
        if self.spec.task.type == "reasoning":
            return "reasoning"
        return "generate"

    def build(self, sample: SampleRecord) -> ModelRequest:
        """Build one stable request containing complete prompt provenance."""
        rendered = self.task.render(sample, self.spec.model.family)
        mode = self._mode_for(sample)
        options = (
            sample.choices
            if mode == "score_options" and isinstance(sample, MultipleChoiceRecord)
            else ()
        )
        prefix = sample.prefix if isinstance(sample, InfillingRecord) else None
        suffix = sample.suffix if isinstance(sample, InfillingRecord) else None
        identity = {
            "run_id": self.spec.run_id,
            "sample_id": sample.sample_id,
            "task_id": self.spec.task.id,
            "model_id": self.spec.model.id,
            "prompt_hash": rendered.prompt_hash,
            "mode": mode,
            "seed": self.spec.seed,
        }
        request_id = f"req-{hash_json(identity)[:24]}"
        return ModelRequest(
            request_id=request_id,
            sample_id=sample.sample_id,
            task_id=self.spec.task.id,
            mode=mode,
            prompt=rendered.text,
            max_output_tokens=self.spec.task.max_output_tokens,
            seed=self.spec.seed,
            options=options,
            prefix=prefix,
            suffix=suffix,
            metadata={
                "request_builder_version": self.VERSION,
                "prompt_id": rendered.prompt_id,
                "prompt_version": rendered.prompt_version,
                "prompt_hash": rendered.prompt_hash,
                "template_hash": rendered.template_hash,
                "semantic_hash": rendered.semantic_hash,
                "output_contract": rendered.output_contract,
                "model_family": rendered.model_family,
                "do_sample": self.spec.decoding.do_sample,
                "temperature": self.spec.decoding.temperature,
                "denoising_steps": self.spec.decoding.denoising_steps,
                "unmasking_strategy": self.spec.decoding.unmasking_strategy,
                "schedule": self.spec.decoding.schedule,
                "mask_policy": self.spec.decoding.mask_policy,
            },
        )


__all__ = ["RequestBuilder"]

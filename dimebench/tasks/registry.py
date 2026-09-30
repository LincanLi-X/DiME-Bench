"""Task and prompt discovery with cross-contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import ValidationError

from dimebench.prompts import PromptError, PromptTemplateSpec, load_prompt_spec
from dimebench.schemas.task import TaskSpec
from dimebench.tasks.base import BaseTask, TaskError
from dimebench.tasks.track1_general import GeneralTask
from dimebench.tasks.track2_infilling import InfillingTask
from dimebench.tasks.track3_editing import EditingTask
from dimebench.tasks.track4_reasoning import ReasoningTask


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load one standalone task YAML configuration."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise TaskError(f"cannot read task config {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TaskError(f"invalid YAML in task config {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TaskError(f"task config {source} must contain a mapping")
    try:
        return TaskSpec.model_validate(payload)
    except ValidationError as exc:
        raise TaskError(f"invalid task config {source}: {exc}") from exc


def create_task(spec: TaskSpec, prompt: PromptTemplateSpec) -> BaseTask:
    """Instantiate the track-specific task implementation."""
    task_type: type[BaseTask]
    if spec.track == "track1_general":
        task_type = GeneralTask
    elif spec.track == "track2_infilling":
        task_type = InfillingTask
    elif spec.track == "track3_editing":
        task_type = EditingTask
    else:
        task_type = ReasoningTask
    return task_type(spec, prompt)


class TaskCatalog:
    """Immutable task registry loaded from repository-tracked YAML assets."""

    def __init__(self, tasks: Mapping[str, BaseTask]) -> None:
        self._tasks = dict(tasks)

    @classmethod
    def from_roots(
        cls,
        task_config_root: str | Path,
        prompt_root: str | Path,
    ) -> TaskCatalog:
        prompt_specs: dict[str, PromptTemplateSpec] = {}
        for path in sorted(Path(prompt_root).glob("track*/*.yaml")):
            prompt = load_prompt_spec(path)
            if prompt.id in prompt_specs:
                raise PromptError(f"duplicate prompt id {prompt.id!r}")
            prompt_specs[prompt.id] = prompt
        if not prompt_specs:
            raise PromptError(f"no prompt definitions found under {prompt_root}")
        tasks: dict[str, BaseTask] = {}
        for path in sorted(Path(task_config_root).glob("track*/*.yaml")):
            spec = load_task_spec(path)
            if spec.id in tasks:
                raise TaskError(f"duplicate task id {spec.id!r}")
            try:
                prompt = prompt_specs[spec.prompt_id]
            except KeyError as exc:
                raise TaskError(
                    f"task {spec.id!r} references unknown prompt {spec.prompt_id!r}"
                ) from exc
            tasks[spec.id] = create_task(spec, prompt)
        if not tasks:
            raise TaskError(f"no task configurations found under {task_config_root}")
        return cls(tasks)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))

    @property
    def catalog_hash(self) -> str:
        from dimebench.artifacts.hashing import hash_json

        return hash_json(
            [
                {
                    "task_id": task_id,
                    "contract_hash": self._tasks[task_id].contract_hash,
                }
                for task_id in self.ids
            ]
        )

    def get(self, task_id: str) -> BaseTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            available = ", ".join(self.ids)
            raise TaskError(
                f"unknown task {task_id!r}; available: {available}"
            ) from exc

    def __len__(self) -> int:
        return len(self._tasks)


__all__ = ["TaskCatalog", "create_task", "load_task_spec"]

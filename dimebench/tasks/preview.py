"""Prompt preview helpers for normalized JSONL data."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from dimebench.datasets import NormalizedDataset
from dimebench.schemas.model import ModelFamily
from dimebench.tasks.base import BaseTask, RenderedPrompt, TaskError
from dimebench.tasks.registry import TaskCatalog


def preview_task_file(
    task: BaseTask,
    sample_path: str | Path,
    families: Sequence[ModelFamily],
    *,
    sample_id: str | None = None,
    limit: int | None = None,
) -> tuple[RenderedPrompt, ...]:
    """Render selected normalized samples without running a model."""
    if limit is not None and limit <= 0:
        raise TaskError("preview limit must be positive")
    dataset = NormalizedDataset.read_jsonl(
        sample_path,
        dataset_id=task.spec.dataset_id,
    )
    selected = [
        record
        for record in dataset
        if sample_id is None or record.sample_id == sample_id
    ]
    if sample_id is not None and not selected:
        raise TaskError(f"sample_id {sample_id!r} was not found in {sample_path}")
    if limit is not None:
        selected = selected[:limit]
    return tuple(
        task.render(record, family) for record in selected for family in families
    )


def preview_prepared_suite(
    catalog: TaskCatalog,
    suite_root: str | Path,
) -> tuple[RenderedPrompt, ...]:
    """Render both model-family prompts for every record in a prepared suite."""
    root = Path(suite_root)
    manifest_path = root / "suite-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskError(f"cannot read suite manifest {manifest_path}: {exc}") from exc
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise TaskError(f"suite manifest {manifest_path} has invalid datasets")
    prompts: list[RenderedPrompt] = []
    families: tuple[ModelFamily, ...] = ("autoregressive", "diffusion")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TaskError("suite manifest dataset entries must be objects")
        dataset_id = entry.get("dataset_id")
        data_path = entry.get("data_path")
        if not isinstance(dataset_id, str) or not isinstance(data_path, str):
            raise TaskError("suite manifest entry lacks dataset_id or data_path")
        task = catalog.get(dataset_id)
        prompts.extend(preview_task_file(task, root / data_path, families))
    return tuple(prompts)


def write_prompt_previews(
    path: str | Path,
    prompts: Iterable[RenderedPrompt],
) -> None:
    """Atomically write deterministic prompt previews as JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            for prompt in prompts:
                payload = json.dumps(
                    prompt.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "preview_prepared_suite",
    "preview_task_file",
    "write_prompt_previews",
]

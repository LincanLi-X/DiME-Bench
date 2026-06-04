from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Any

from src.utils.config import resolve_path
from src.utils.io import read_jsonl


def load_and_normalize_track6_dataset(
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    *,
    allow_synthetic_fallback: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset_name == "fixed_length":
        rows = _fixed_length_samples(dataset_cfg)
        return rows, {"source_files": [], "used_synthetic_fallback": False}

    raw_rows, source_files = _load_upstream_rows(dataset_cfg)
    used_fallback = False
    if not raw_rows:
        if not allow_synthetic_fallback:
            raise FileNotFoundError(
                f"No upstream JSONL files found for {dataset_name}. "
                "Pass --allow-synthetic-fallback only for smoke tests."
            )
        raw_rows = _fallback_rows(dataset_name)
        used_fallback = True

    normalized = [
        _normalize_row(dataset_name, dataset_cfg, idx, row, used_fallback=used_fallback)
        for idx, row in enumerate(raw_rows)
    ]
    return normalized, {"source_files": source_files, "used_synthetic_fallback": used_fallback}


def select_track6_samples(
    rows: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
    *,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_seed = int(sample_seed if sample_seed is not None else dataset_cfg.get("sample_seed", 42))
    threshold = int(dataset_cfg.get("sample_threshold", 380))
    configured_size = dataset_cfg.get("sample_size")
    if sample_size is None and configured_size is not None and len(rows) > threshold:
        sample_size = int(configured_size)

    selection: dict[str, Any] = {
        "source_num_samples": len(rows),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "mode": "full",
    }
    selected = list(rows)
    if sample_size is not None and len(rows) > sample_size:
        rng = random.Random(sample_seed)
        indices = sorted(rng.sample(range(len(rows)), sample_size))
        selected = [rows[idx] for idx in indices]
        selection.update({"mode": "random_sample", "selected_indices": indices})
    if limit is not None:
        selected = selected[:limit]
        selection["limit"] = limit
        selection["mode"] = selection["mode"] + "+limit"
    return selected, selection


def _load_upstream_rows(dataset_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    source_files: list[str] = []
    for pattern in dataset_cfg.get("upstream_paths", []):
        resolved_pattern = str(resolve_path(pattern))
        for path_str in sorted(glob.glob(resolved_pattern, recursive=True)):
            path = Path(path_str)
            if not path.is_file() or path_str in source_files:
                continue
            file_rows = read_jsonl(path)
            if file_rows:
                rows.extend(file_rows)
                source_files.append(str(path))
    return rows, source_files


def _normalize_row(
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    idx: int,
    row: dict[str, Any],
    *,
    used_fallback: bool,
) -> dict[str, Any]:
    if dataset_name == "cnn_dailymail_infilling":
        prefix = _first(row, "prefix", "left_context", "context_before", "article_prefix")
        suffix = _first(row, "suffix", "right_context", "context_after", "article_suffix")
        reference = _first(row, "reference", "missing", "target", "gold", "middle", "answer")
        prompt_fields = {"prefix": prefix, "suffix": suffix}
    elif dataset_name == "synthetic_contradiction_repair":
        text = _first(row, "text", "input", "contradictory_text", "source_text", "original")
        instruction = _first(row, "instruction", "edit_instruction", "repair_instruction")
        reference = _first(row, "reference", "target", "gold", "repaired", "answer")
        prompt_fields = {"text": text, "instruction": instruction or "Remove the contradiction."}
    elif dataset_name == "json_schema_generation":
        task = _first(row, "task", "instruction", "prompt", "description")
        schema = row.get("schema") or row.get("json_schema") or row.get("target_schema") or {}
        reference = row.get("reference") or row.get("target") or row.get("gold") or {}
        prompt_fields = {"task": task, "schema": _stringify_schema(schema)}
    else:
        raise ValueError(f"Unsupported Track 6 dataset: {dataset_name}")

    return {
        "sample_id": str(row.get("sample_id") or row.get("id") or f"{dataset_name}_{idx:06d}"),
        "dataset": dataset_name,
        "task_type": dataset_cfg["task_type"],
        "prompt_fields": prompt_fields,
        "reference": reference,
        "target_output_length": int(row.get("target_output_length") or dataset_cfg.get("max_new_tokens", 512)),
        "max_new_tokens": int(row.get("max_new_tokens") or dataset_cfg.get("max_new_tokens", 512)),
        "source": "synthetic_fallback" if used_fallback else "upstream_processed",
        "raw": row,
    }


def _fixed_length_samples(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    topics = [
        "urban water planning",
        "open source software maintenance",
        "renewable energy storage",
        "clinical trial coordination",
    ]
    rows = []
    for target_length in dataset_cfg.get("target_lengths", [128, 512, 1024]):
        for idx, topic in enumerate(topics):
            rows.append(
                {
                    "sample_id": f"fixed_length_{target_length}_{idx}",
                    "dataset": "fixed_length",
                    "task_type": "fixed_length",
                    "prompt_fields": {
                        "topic": topic,
                        "target_output_length": int(target_length),
                    },
                    "reference": {"target_output_length": int(target_length), "topic": topic},
                    "target_output_length": int(target_length),
                    "max_new_tokens": int(target_length),
                    "source": "synthetic_control",
                    "raw": {},
                }
            )
    return rows


def _fallback_rows(dataset_name: str) -> list[dict[str, Any]]:
    if dataset_name == "cnn_dailymail_infilling":
        return [
            {
                "sample_id": "fallback_infilling_0",
                "prefix": "The city council approved the emergency plan after a week of public hearings.",
                "missing": "Officials said the first phase would focus on shelter capacity and food distribution.",
                "suffix": "The mayor said the program would be reviewed again next month.",
            },
            {
                "sample_id": "fallback_infilling_1",
                "prefix": "A regional hospital expanded its mobile clinic program on Monday.",
                "missing": "The new vans will visit rural communities twice a week and offer basic screenings.",
                "suffix": "Administrators expect the service to reduce missed appointments.",
            },
        ]
    if dataset_name == "synthetic_contradiction_repair":
        return [
            {
                "sample_id": "fallback_repair_0",
                "text": "Maya moved to Denver in 2019. Later the same paragraph says Maya has never lived outside Boston.",
                "instruction": "Resolve the contradiction while preserving the timeline.",
                "reference": "Maya moved to Denver in 2019 after living in Boston.",
            },
            {
                "sample_id": "fallback_repair_1",
                "text": "The lab opened in April. The summary then states the lab remained closed all spring.",
                "instruction": "Make the status consistent.",
                "reference": "The lab opened in April and operated through the rest of spring.",
            },
        ]
    if dataset_name == "json_schema_generation":
        return [
            {
                "sample_id": "fallback_json_0",
                "task": "Create a short user profile.",
                "schema": {
                    "type": "object",
                    "required": ["name", "role", "active"],
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "active": {"type": "boolean"},
                    },
                },
                "reference": {"name": "Alex", "role": "analyst", "active": True},
            },
            {
                "sample_id": "fallback_json_1",
                "task": "Create an inventory item.",
                "schema": {
                    "type": "object",
                    "required": ["sku", "quantity"],
                    "properties": {
                        "sku": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                },
                "reference": {"sku": "A-100", "quantity": 3},
            },
        ]
    raise ValueError(f"No fallback rows defined for {dataset_name}")


def _first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _stringify_schema(schema: Any) -> str:
    if isinstance(schema, str):
        return schema
    import json

    return json.dumps(schema, ensure_ascii=False, sort_keys=True)

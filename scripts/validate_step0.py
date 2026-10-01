#!/usr/bin/env python3
"""Validate the frozen DiME-Bench v1 protocol using only the standard library."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_TRACKS = {
    "track1_general",
    "track2_infilling",
    "track3_editing",
    "track4_reasoning",
}
SPECIAL_TASK_SCOPES = {"__track__", "__dataset_row__"}


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    spec_path = root / "specification" / "benchmark-v1.json"

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"missing specification: {spec_path}"]
    except json.JSONDecodeError as exc:
        return [f"invalid JSON specification: {exc}"]

    if spec.get("schema_version") != "1.0.0":
        errors.append("schema_version must be 1.0.0")
    if spec.get("benchmark", {}).get("version") != "1.0.0":
        errors.append("benchmark version must be 1.0.0")

    tracks = spec.get("tracks", [])
    track_ids = {track.get("id") for track in tracks}
    if track_ids != EXPECTED_TRACKS:
        errors.append(
            f"expected tracks {sorted(EXPECTED_TRACKS)}, got {sorted(track_ids)}"
        )

    track_map = {track["id"]: track for track in tracks if "id" in track}
    for track_id, track in track_map.items():
        task_ids = [task.get("id") for task in track.get("tasks", [])]
        if not task_ids or any(not task_id for task_id in task_ids):
            errors.append(f"{track_id}: every task must have a non-empty id")
        if len(task_ids) != len(set(task_ids)):
            errors.append(f"{track_id}: duplicate task ids")
        metric_ids = set(track.get("metric_ids", []))
        if not metric_ids:
            errors.append(f"{track_id}: metric_ids must not be empty")
        for task in track.get("tasks", []):
            if task.get("primary_metric") not in metric_ids:
                errors.append(
                    f"{track_id}/{task.get('id')}: primary metric is not registered"
                )

    seen_columns: set[tuple[str, str]] = set()
    for mapping in spec.get("paper_result_columns", []):
        key = (mapping.get("table", ""), mapping.get("column", ""))
        if key in seen_columns:
            errors.append(f"duplicate paper column mapping: {key}")
        seen_columns.add(key)

        track_id = mapping.get("track")
        track = track_map.get(track_id)
        if track is None:
            errors.append(f"paper mapping references unknown track: {track_id}")
            continue
        if mapping.get("metric") not in set(track.get("metric_ids", [])):
            errors.append(f"paper mapping {key} references unknown metric")
        task_id = mapping.get("task")
        valid_tasks = {task.get("id") for task in track.get("tasks", [])}
        if task_id not in valid_tasks | SPECIAL_TASK_SCOPES:
            errors.append(f"paper mapping {key} references unknown task: {task_id}")

    mapped_tables = {
        mapping.get("table") for mapping in spec.get("paper_result_columns", [])
    }
    if mapped_tables != {"Table 2", "Table 3", "Table 4", "Table 5"}:
        errors.append("paper mappings must cover Tables 2, 3, 4, and 5")

    for relative_path in spec.get("required_documents", []):
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing required document: {relative_path}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"empty required document: {relative_path}")

    required_defaults = {
        "deterministic",
        "dllm_denoising_steps",
        "dllm_unmasking",
        "ar_decoding",
        "retain_failures",
    }
    missing_defaults = required_defaults - set(spec.get("defaults", {}))
    if missing_defaults:
        errors.append(f"missing comparison defaults: {sorted(missing_defaults)}")

    if spec.get("defaults", {}).get("dllm_denoising_steps") != 128:
        errors.append("default dLLM denoising steps must be 128")
    if not spec.get("defaults", {}).get("retain_failures"):
        errors.append("failure retention must be enabled")
    if not spec.get("failure_classes"):
        errors.append("failure_classes must not be empty")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = validate(root)
    if errors:
        print("Step 0 validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Step 0 validation passed")
    print("- benchmark protocol: 1.0.0")
    print("- tracks: 4")
    print("- paper tables mapped: 2, 3, 4, 5")
    print("- required specification documents: 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())

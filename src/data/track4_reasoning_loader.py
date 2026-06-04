from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.utils.config import resolve_path
from src.utils.io import read_jsonl


OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_raw_track4_dataset(dataset_name: str, dataset_cfg: dict[str, Any], cache_dir: str) -> list[dict[str, Any]]:
    if dataset_name == "gsm8k":
        track1_path = resolve_path("data/processed/track1_general/gsm8k/samples.jsonl")
        if track1_path.exists():
            return read_jsonl(track1_path)
    if dataset_name == "path_star":
        return generate_path_star_rows(dataset_cfg.get("generation_params") or {})

    local_path = dataset_cfg.get("local_path")
    if local_path:
        path = resolve_path(local_path)
        if path.exists():
            return read_jsonl(path)

    raw = _load_hf_dataset(
        repo_id=dataset_cfg["repo_id"],
        subset=dataset_cfg.get("subset"),
        split=dataset_cfg["split"],
        cache_dir=cache_dir,
    )
    return [dict(row) for row in raw]


def normalize_track4_rows(dataset_name: str, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if dataset_name == "gsm8k":
        return [_normalize_gsm8k(i, row) for i, row in enumerate(raw_rows)]
    if dataset_name == "math_500":
        return [_normalize_math500(i, row) for i, row in enumerate(raw_rows)]
    if dataset_name == "winogrande":
        return [_normalize_winogrande(i, row) for i, row in enumerate(raw_rows)]
    if dataset_name == "path_star":
        return [_normalize_path_star(i, row) for i, row in enumerate(raw_rows)]
    raise ValueError(f"Unsupported Track 4 dataset: {dataset_name}")


def generate_path_star_rows(params: dict[str, Any]) -> list[dict[str, Any]]:
    num_samples = int(params.get("num_samples", 500))
    num_arms = int(params.get("num_arms", 5))
    arm_length = int(params.get("arm_length", 4))
    seed = int(params.get("seed", 42))
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for sample_idx in range(num_samples):
        start = f"s{sample_idx}"
        nodes = [f"n{sample_idx}_{i}" for i in range(num_arms * arm_length)]
        rng.shuffle(nodes)
        arms = []
        for arm_idx in range(num_arms):
            arm_nodes = nodes[arm_idx * arm_length : (arm_idx + 1) * arm_length]
            arms.append([start, *arm_nodes])
        target_arm = rng.randrange(num_arms)
        reference_path = arms[target_arm]
        rows.append(
            {
                "sample_id": str(sample_idx),
                "dataset": "path_star",
                "arms": arms,
                "start": start,
                "target": reference_path[-1],
                "answer": reference_path,
                "generation_params": {
                    "num_arms": num_arms,
                    "arm_length": arm_length,
                    "seed": seed,
                },
            }
        )
    return rows


def _load_hf_dataset(repo_id: str, subset: str | None, split: str, cache_dir: str) -> Dataset:
    if subset:
        return load_dataset(repo_id, subset, split=split, cache_dir=cache_dir)
    return load_dataset(repo_id, split=split, cache_dir=cache_dir)


def _normalize_gsm8k(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer", ""))
    final = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
    return {
        "sample_id": str(row.get("sample_id", idx)),
        "dataset": "gsm8k",
        "task_type": "math",
        "reasoning_type": "chaining",
        "question": str(row["question"]),
        "answer": final,
        "reference": final,
        "rationale": row.get("rationale") or answer,
        "raw": dict(row),
    }


def _normalize_math500(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    problem = row.get("problem") or row.get("question") or row.get("prompt")
    answer = row.get("answer") or row.get("final_answer")
    solution = row.get("solution") or row.get("rationale")
    if problem is None or answer is None:
        raise ValueError(f"MATH-500 row {idx} missing problem/answer fields: {sorted(row.keys())}")
    return {
        "sample_id": str(row.get("unique_id") or row.get("id") or idx),
        "dataset": "math_500",
        "task_type": "math",
        "reasoning_type": "chaining",
        "question": str(problem),
        "answer": str(answer).strip(),
        "reference": str(answer).strip(),
        "solution": str(solution or ""),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "raw": dict(row),
    }


def _normalize_winogrande(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip()
    if answer in {"0", "1"}:
        answer_letter = "A"
    elif answer == "2":
        answer_letter = "B"
    else:
        answer_letter = answer.upper()[:1]
    return {
        "sample_id": str(row.get("qID") or row.get("id") or idx),
        "dataset": "winogrande",
        "task_type": "multiple_choice",
        "reasoning_type": "planning",
        "sentence": str(row["sentence"]),
        "options": [str(row["option1"]), str(row["option2"])],
        "answer": answer_letter,
        "reference": answer_letter,
        "raw": dict(row),
    }


def _normalize_path_star(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    arms = row.get("arms") or row.get("graph") or []
    if isinstance(arms, str):
        arms = _parse_arms_text(arms)
    reference_path = row.get("answer") or row.get("reference_path")
    if isinstance(reference_path, str):
        reference_path = parse_path_text(reference_path)
    if not arms or not reference_path:
        raise ValueError(f"Path-Star row {idx} missing arms/reference path.")
    start = row.get("start") or reference_path[0]
    target = row.get("target") or reference_path[-1]
    return {
        "sample_id": str(row.get("sample_id", idx)),
        "dataset": "path_star",
        "task_type": "graph_planning",
        "reasoning_type": "planning",
        "arms": [[str(node) for node in arm] for arm in arms],
        "graph_text": graph_to_text(arms),
        "start": str(start),
        "target": str(target),
        "answer": [str(node) for node in reference_path],
        "reference": [str(node) for node in reference_path],
        "raw": dict(row),
    }


def graph_to_text(arms: list[list[str]]) -> str:
    return "\n".join(f"Arm {idx + 1}: {' -> '.join(map(str, arm))}" for idx, arm in enumerate(arms))


def parse_path_text(text: str) -> list[str]:
    cleaned = str(text).replace(",", " -> ").replace(";", " -> ")
    parts = [part.strip() for part in cleaned.split("->")]
    if len(parts) == 1:
        parts = [part.strip() for part in cleaned.split()]
    return [part for part in parts if part]


def _parse_arms_text(text: str) -> list[list[str]]:
    arms = []
    for line in str(text).splitlines():
        if "->" not in line:
            continue
        _, _, path = line.partition(":")
        arms.append(parse_path_text(path or line))
    return arms


def processed_track4_path(dataset_name: str) -> Path:
    return resolve_path(Path("data/processed/track4_reasoning") / dataset_name / "samples.jsonl")

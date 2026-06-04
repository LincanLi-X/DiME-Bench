from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, load_dataset


OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_hf_dataset(repo_id: str, subset: str | None, split: str, cache_dir: str) -> Dataset:
    if subset:
        return load_dataset(repo_id, subset, split=split, cache_dir=cache_dir)
    return load_dataset(repo_id, split=split, cache_dir=cache_dir)


def load_raw_track1_dataset(dataset_cfg: dict[str, Any], cache_dir: str) -> Dataset:
    return _load_hf_dataset(
        repo_id=dataset_cfg["repo_id"],
        subset=dataset_cfg.get("subset"),
        split=dataset_cfg["split"],
        cache_dir=cache_dir,
    )


def normalize_track1_rows(dataset_name: str, raw: Dataset) -> list[dict[str, Any]]:
    if dataset_name == "mmlu_pro":
        return [_normalize_mmlu_pro(i, row) for i, row in enumerate(raw)]
    if dataset_name == "hellaswag":
        return [_normalize_hellaswag(i, row) for i, row in enumerate(raw)]
    if dataset_name == "gsm8k":
        return [_normalize_gsm8k(i, row) for i, row in enumerate(raw)]
    if dataset_name == "humaneval":
        return [_normalize_humaneval(i, row) for i, row in enumerate(raw)]
    if dataset_name == "ifeval":
        return [_normalize_ifeval(i, row) for i, row in enumerate(raw)]
    raise ValueError(f"Unsupported Track 1 dataset: {dataset_name}")


def _coerce_options(options: Any) -> list[str]:
    if isinstance(options, list):
        return [str(x) for x in options]
    if isinstance(options, tuple):
        return [str(x) for x in options]
    if isinstance(options, str):
        # MMLU-Pro stores list-like strings in some mirrors.
        stripped = options.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [part.strip() for part in re.split(r"\s*\|\s*", options) if part.strip()]
    raise TypeError(f"Cannot coerce options from {type(options)}")


def _answer_to_letter(answer: Any, options: list[str]) -> str:
    if isinstance(answer, int):
        return OPTION_LETTERS[answer]
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in OPTION_LETTERS:
        return text.upper()
    if text.isdigit():
        return OPTION_LETTERS[int(text)]
    for idx, opt in enumerate(options):
        if text == opt or text.lower() == opt.lower():
            return OPTION_LETTERS[idx]
    return text


def _normalize_mmlu_pro(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    options = _coerce_options(row.get("options") or row.get("choices"))
    answer = row.get("answer")
    if answer is None:
        answer = row.get("answer_index")
    answer_letter = _answer_to_letter(answer, options)
    return {
        "sample_id": str(row.get("question_id", idx)),
        "dataset": "mmlu_pro",
        "task_type": "multiple_choice",
        "question": str(row["question"]),
        "options": options,
        "answer": answer_letter,
        "category": row.get("category") or row.get("subject"),
        "raw": dict(row),
    }


def _normalize_hellaswag(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    options = _coerce_options(row["endings"])
    answer_letter = _answer_to_letter(row.get("label"), options)
    context = (str(row.get("ctx_a", "")) + " " + str(row.get("ctx_b", ""))).strip()
    if not context:
        context = str(row.get("ctx", ""))
    return {
        "sample_id": str(row.get("ind", idx)),
        "dataset": "hellaswag",
        "task_type": "multiple_choice",
        "question": context,
        "activity_label": row.get("activity_label"),
        "options": options,
        "answer": answer_letter,
        "raw": dict(row),
    }


def _normalize_gsm8k(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row["answer"])
    final = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
    return {
        "sample_id": str(idx),
        "dataset": "gsm8k",
        "task_type": "math",
        "question": str(row["question"]),
        "answer": final,
        "rationale": answer,
        "raw": dict(row),
    }


def _normalize_humaneval(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row.get("task_id", idx)),
        "dataset": "humaneval",
        "task_type": "code",
        "prompt": str(row["prompt"]),
        "entry_point": str(row["entry_point"]),
        "test": str(row["test"]),
        "canonical_solution": str(row.get("canonical_solution", "")),
        "raw": dict(row),
    }


def _normalize_ifeval(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row.get("key", idx)),
        "dataset": "ifeval",
        "task_type": "instruction_following",
        "prompt": str(row["prompt"]),
        "instruction_id_list": list(row.get("instruction_id_list", [])),
        "kwargs": row.get("kwargs", []),
        "raw": dict(row),
    }

from __future__ import annotations

import csv
import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from src.metrics.track3_editing import diff_target_token_indices, edit_distance, tokenize_text
from src.utils.config import resolve_path
from src.utils.io import read_jsonl


def load_raw_track3_dataset(dataset_name: str, dataset_cfg: dict[str, Any], cache_dir: str) -> list[dict[str, Any]]:
    if dataset_name == "synthetic_contradiction_repair":
        return generate_synthetic_contradiction_rows(int(dataset_cfg.get("synthetic_size", 240)))

    for raw_path in dataset_cfg.get("raw_paths", []):
        path = resolve_path(raw_path)
        if path.exists():
            return _read_table(path)

    hf_cfg = dataset_cfg.get("hf") or {}
    repo_id = hf_cfg.get("repo_id")
    if repo_id:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("datasets is not installed; provide a local raw file for Track 3.") from exc
        subset = hf_cfg.get("subset")
        split = hf_cfg.get("split", "test")
        dataset = (
            load_dataset(repo_id, subset, split=split, cache_dir=cache_dir)
            if subset
            else load_dataset(repo_id, split=split, cache_dir=cache_dir)
        )
        return [dict(row) for row in dataset]

    expected = ", ".join(dataset_cfg.get("raw_paths", []))
    raise FileNotFoundError(
        f"No local raw file found for {dataset_name}. Expected one of: {expected}. "
        "Set datasets.<name>.hf.repo_id in configs/tracks/track3_editing.yaml if an accessible HF mirror is available."
    )


def normalize_track3_rows(dataset_name: str, raw: list[dict[str, Any]], dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if dataset_name == "conll2014_gec":
        return [_normalize_conll(idx, row, dataset_cfg) for idx, row in enumerate(raw)]
    if dataset_name == "fruit":
        return [_normalize_fruit(idx, row, dataset_cfg) for idx, row in enumerate(raw)]
    if dataset_name == "synthetic_contradiction_repair":
        return [_finalize_sample(row) for row in raw]
    raise ValueError(f"Unsupported Track 3 dataset: {dataset_name}")


def load_processed_track3_samples(dataset_name: str) -> list[dict[str, Any]]:
    path = resolve_path(Path("data/processed/track3_editing") / dataset_name / "samples.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run scripts/prepare_track3_editing_datasets.py first."
        )
    return read_jsonl(path)


def generate_synthetic_contradiction_rows(size: int = 240) -> list[dict[str, Any]]:
    templates = [
        {
            "name": "city_residence",
            "source": "{person} moved to {city} in {year}. In a later profile, the article says {person} had never lived in {city}.",
            "reference": "{person} moved to {city} in {year}. In a later profile, the article says {person} lived in {city}.",
            "instruction": "Repair the contradiction about whether {person} lived in {city}.",
            "target": "had never lived in {city}",
            "required": "lived in {city}",
            "forbidden": "never lived in {city}",
        },
        {
            "name": "award_winner",
            "source": "{person} won the {award} in {year}. The summary later states that {person} did not win the {award}.",
            "reference": "{person} won the {award} in {year}. The summary later states that {person} won the {award}.",
            "instruction": "Repair the contradiction about the {award} result.",
            "target": "did not win the {award}",
            "required": "won the {award}",
            "forbidden": "did not win the {award}",
        },
        {
            "name": "company_founder",
            "source": "{person} founded {company} with two colleagues. The next sentence claims {person} was not a founder of {company}.",
            "reference": "{person} founded {company} with two colleagues. The next sentence claims {person} was a founder of {company}.",
            "instruction": "Repair the contradiction about {person}'s role at {company}.",
            "target": "was not a founder of {company}",
            "required": "was a founder of {company}",
            "forbidden": "was not a founder of {company}",
        },
        {
            "name": "publication_date",
            "source": "The report was published in {year}. A caption below says the report was not published in {year}.",
            "reference": "The report was published in {year}. A caption below says the report was published in {year}.",
            "instruction": "Repair the contradiction about the publication year.",
            "target": "was not published in {year}",
            "required": "was published in {year}",
            "forbidden": "was not published in {year}",
        },
    ]
    people = ["Maya Chen", "Noah Patel", "Elena Garcia", "Omar Reed", "Lina Brooks"]
    cities = ["Paris", "Berlin", "Toronto", "Seoul", "Lisbon"]
    awards = ["Arbor Prize", "Civic Medal", "Kepler Award", "North Star Grant"]
    companies = ["Helio Labs", "Riverbyte", "Cedar Analytics", "Brightline Systems"]
    years = ["2016", "2018", "2020", "2021", "2023"]
    rows: list[dict[str, Any]] = []
    for idx in range(size):
        template = templates[idx % len(templates)]
        values = {
            "person": people[idx % len(people)],
            "city": cities[(idx // 2) % len(cities)],
            "award": awards[(idx // 3) % len(awards)],
            "company": companies[(idx // 4) % len(companies)],
            "year": years[(idx // 5) % len(years)],
        }
        source = template["source"].format(**values)
        reference = template["reference"].format(**values)
        target_text = template["target"].format(**values)
        rows.append(
            {
                "sample_id": f"synthetic-{idx:04d}",
                "dataset": "synthetic_contradiction_repair",
                "task_type": "contradiction_repair",
                "source_text": source,
                "instruction": template["instruction"].format(**values),
                "reference": reference,
                "evidence": "",
                "target_span_text": target_text,
                "edit_type": template["name"],
                "required_phrases": [template["required"].format(**values)],
                "forbidden_phrases": [template["forbidden"].format(**values)],
                "contradiction_before": [template["forbidden"].format(**values)],
                "contradiction_after": [],
                "contradiction_count_before": 1,
            }
        )
    return rows


def _normalize_conll(idx: int, row: dict[str, Any], dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    source = _pick(row, dataset_cfg["required_fields"]["source_text"])
    reference = _pick(row, dataset_cfg["required_fields"]["reference"])
    sample = {
        "sample_id": str(row.get("sample_id") or row.get("id") or idx),
        "dataset": "conll2014_gec",
        "task_type": "grammatical_error_correction",
        "source_text": source,
        "instruction": "Correct grammar, spelling, and word-choice errors while preserving meaning.",
        "reference": reference,
        "evidence": "",
        "edit_type": "gec",
        "raw": row,
    }
    return _finalize_sample(sample)


def _normalize_fruit(idx: int, row: dict[str, Any], dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    required = dataset_cfg["required_fields"]
    source = _pick(row, required["source_text"])
    reference = _pick(row, required["reference"])
    evidence = _pick(row, required.get("evidence", []), default=str(row.get("instruction") or row.get("update") or ""))
    instruction = str(
        row.get("instruction")
        or row.get("edit_instruction")
        or row.get("update_instruction")
        or "Update the source text so it is consistent with the evidence."
    )
    sample = {
        "sample_id": str(row.get("sample_id") or row.get("id") or idx),
        "dataset": "fruit",
        "task_type": "evidence_grounded_editing",
        "source_text": source,
        "instruction": instruction,
        "reference": reference,
        "evidence": evidence,
        "edit_type": str(row.get("edit_type") or "evidence_update"),
        "contradiction_count_before": int(row.get("contradiction_count_before") or 0),
        "forbidden_phrases": _coerce_list(row.get("forbidden_phrases")),
        "required_phrases": _coerce_list(row.get("required_phrases")),
        "raw": row,
    }
    return _finalize_sample(sample)


def _finalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    source = sample["source_text"]
    reference = sample["reference"]
    source_tokens = tokenize_text(source)
    reference_tokens = tokenize_text(reference)
    target_indices = sample.get("target_token_indices") or diff_target_token_indices(source, reference)
    target_text = sample.get("target_span_text") or _target_text_from_indices(source_tokens, target_indices)
    minimal_edit_distance = sample.get("minimal_edit_distance")
    if minimal_edit_distance is None:
        minimal_edit_distance = edit_distance(source_tokens, reference_tokens)
    sample.update(
        {
            "target_span": {
                "source_token_indices": target_indices,
                "source_text": target_text,
            },
            "target_token_indices": target_indices,
            "non_target_spans": _non_target_spans(len(source_tokens), set(target_indices)),
            "minimal_edit_distance": int(minimal_edit_distance),
        }
    )
    if "marked_source_text" not in sample:
        sample["marked_source_text"] = _mark_target_span(source_tokens, set(target_indices))
    return sample


def _read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [dict(row) for row in data]
        if isinstance(data, dict):
            for key in ("data", "samples", "examples"):
                if isinstance(data.get(key), list):
                    return [dict(row) for row in data[key]]
        raise ValueError(f"Unsupported JSON shape in {path}")
    dialect = "excel-tab" if suffix in {".tsv", ".tab"} else "excel"
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, dialect=dialect)]


def _pick(row: dict[str, Any], names: Iterable[str], default: str | None = None) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value)
    if default is not None:
        return default
    raise KeyError(f"Could not find any of fields {list(names)} in row keys {sorted(row.keys())}")


def _coerce_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [part.strip() for part in value.split("|") if part.strip()]
    return [str(value)]


def _target_text_from_indices(tokens: list[str], indices: list[int]) -> str:
    if not indices:
        return ""
    return " ".join(tokens[min(indices) : max(indices) + 1])


def _non_target_spans(num_tokens: int, target_indices: set[int]) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    start: int | None = None
    for idx in range(num_tokens):
        if idx in target_indices:
            if start is not None:
                spans.append({"start_token": start, "end_token": idx})
                start = None
        elif start is None:
            start = idx
    if start is not None:
        spans.append({"start_token": start, "end_token": num_tokens})
    return spans


def _mark_target_span(tokens: list[str], target_indices: set[int]) -> str:
    if not target_indices:
        return " ".join(tokens)
    pieces: list[str] = []
    in_target = False
    for idx, token in enumerate(tokens):
        target = idx in target_indices
        if target and not in_target:
            pieces.append("<edit>")
            in_target = True
        if not target and in_target:
            pieces.append("</edit>")
            in_target = False
        pieces.append(token)
    if in_target:
        pieces.append("</edit>")
    return " ".join(pieces)


def select_track3_samples(
    samples: list[dict[str, Any]],
    sample_size: int | None,
    sample_seed: int,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection: dict[str, Any] = {
        "source_num_samples": len(samples),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "mode": "full",
    }
    selected = samples
    if sample_size is not None and len(selected) > sample_size:
        rng = random.Random(sample_seed)
        selected_indices = sorted(rng.sample(range(len(selected)), sample_size))
        selected = [selected[idx] for idx in selected_indices]
        selection.update({"mode": "random_sample", "selected_indices": selected_indices})
    if limit is not None:
        selected = selected[:limit]
        selection["limit"] = limit
        selection["mode"] = selection["mode"] + "+limit"
    return selected, selection

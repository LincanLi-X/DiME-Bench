"""Deterministic raw-field normalization and suite preparation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from dimebench.artifacts.hashing import write_json_atomic
from dimebench.datasets.base import (
    DatasetError,
    DatasetManifest,
    NormalizedDataset,
    load_data_registry,
    load_dataset_manifest,
)
from dimebench.datasets.download import (
    acquire_full_source,
    load_bundled_sample_rows,
    read_jsonl_rows,
)
from dimebench.datasets.records import (
    SampleRecord,
    TextSpan,
    expected_sample_id,
    parse_sample_record,
)
from dimebench.datasets.split_lock import freeze_split, write_split_lock


def _field(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    current: Any = row
    for part in name.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _mapped(
    row: Mapping[str, Any],
    mapping: Mapping[str, str],
    target: str,
    default: Any = None,
) -> Any:
    return _field(row, mapping.get(target, target), default)


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    raise DatasetError(f"expected a string or sequence of strings, got {type(value)}")


def _int_parameter(parameters: Mapping[str, JsonValue], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if isinstance(value, (bool, int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"preprocessing parameter {name!r} must be an integer"
            ) from exc
    raise DatasetError(f"preprocessing parameter {name!r} must be an integer")


def _source_id(row: Mapping[str, Any], index: int) -> str:
    for field_name in ("source_id", "id", "task_id", "question_id", "index"):
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value)
    return f"row-{index:08d}"


def _metadata(row: Mapping[str, Any]) -> dict[str, JsonValue]:
    value = row.get("metadata", {})
    if not isinstance(value, Mapping):
        raise DatasetError("sample metadata must be a JSON object")
    return cast(dict[str, JsonValue], dict(value))


def _with_identity(
    manifest: DatasetManifest,
    split: str,
    source_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "sample_id": f"{manifest.id}.{split}.pending",
        "dataset_id": manifest.id,
        "split": split,
        "source_id": source_id,
        **payload,
    }


def _span_length_bin(length: int) -> str:
    if length <= 16:
        return "short"
    if length <= 64:
        return "medium"
    return "long"


def _boundary_density(prefix: str, suffix: str) -> str:
    boundary = prefix.split()[-12:] + suffix.split()[:12]
    informative = sum(
        token[:1].isupper() or any(character.isdigit() for character in token)
        for token in boundary
    )
    return "high" if informative >= 2 else "low"


def _derive_infilling_parts(text: str) -> tuple[str, str, str]:
    words = text.split()
    if len(words) < 9:
        raise DatasetError("infilling source text requires at least 9 words")
    start = max(2, len(words) // 3)
    end = min(len(words) - 2, max(start + 1, (2 * len(words)) // 3))
    return " ".join(words[:start]), " ".join(words[start:end]), " ".join(words[end:])


def _generation_references(
    row: Mapping[str, Any],
    mapping: Mapping[str, str],
    parameters: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    references = _strings(_mapped(row, mapping, "references"))
    extractor = parameters.get("reference_extractor")
    if extractor is None:
        return references
    if extractor != "gsm8k_final_answer":
        raise DatasetError(f"unknown generation reference extractor {extractor!r}")
    extracted: list[str] = []
    for reference in references:
        match = re.search(r"####\s*([^\n]+)\s*$", reference)
        if match is None:
            raise DatasetError("GSM8K reference is missing its final #### answer")
        extracted.append(match.group(1).strip())
    return tuple(extracted)


def _levenshtein(left: str, right: str) -> int:
    left_tokens = tuple(
        re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", left).casefold())
    )
    right_tokens = tuple(
        re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", right).casefold())
    )
    previous = list(range(len(right_tokens) + 1))
    for left_index, left_token in enumerate(left_tokens, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right_tokens, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _derive_edit_spans(
    source: str,
    reference: str,
) -> tuple[tuple[TextSpan, ...], tuple[TextSpan, ...]]:
    changed_ranges: list[tuple[int, int]] = []
    for tag, source_start, source_end, _, _ in SequenceMatcher(
        None, source, reference, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        if source_start == source_end:
            source_start = max(0, source_start - 1)
            source_end = min(len(source), source_end + 1)
        changed_ranges.append((source_start, source_end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(changed_ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    targets = tuple(
        TextSpan(start=start, end=end, text=source[start:end]) for start, end in merged
    )
    preserved: list[TextSpan] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            preserved.append(
                TextSpan(start=cursor, end=start, text=source[cursor:start])
            )
        cursor = end
    if cursor < len(source):
        preserved.append(TextSpan(start=cursor, end=len(source), text=source[cursor:]))
    return targets, tuple(preserved)


def _normalize_payload(
    manifest: DatasetManifest,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    strategy = manifest.preprocessing.strategy
    mapping = manifest.preprocessing.field_map
    parameters = manifest.preprocessing.parameters
    metadata = _metadata(row)
    if strategy == "normalized" or row.get("kind") is not None:
        payload = dict(row)
        for key in ("schema_version", "sample_id", "dataset_id", "split", "source_id"):
            payload.pop(key, None)
        return payload
    if strategy == "multiple_choice":
        choices = _strings(_mapped(row, mapping, "choices"))
        answer = _mapped(row, mapping, "answer_index")
        if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
            answer = ord(answer.upper()) - ord("A")
        return {
            "kind": "multiple_choice",
            "question": str(_mapped(row, mapping, "question")),
            "choices": choices,
            "answer_index": int(answer),
            "context": _mapped(row, mapping, "context"),
            "category": _mapped(row, mapping, "category"),
            "metadata": metadata,
        }
    if strategy == "generation":
        return {
            "kind": "generation",
            "instruction": str(_mapped(row, mapping, "instruction")),
            "references": _generation_references(row, mapping, parameters),
            "context": _mapped(row, mapping, "context"),
            "tests": _strings(_mapped(row, mapping, "tests", ())),
            "metadata": metadata,
        }
    if strategy == "infilling":
        prefix = _mapped(row, mapping, "prefix")
        middle = _mapped(row, mapping, "middle")
        suffix = _mapped(row, mapping, "suffix")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (prefix, middle, suffix)
        ):
            text = str(_mapped(row, mapping, "text"))
            prefix, middle, suffix = _derive_infilling_parts(text)
        assert isinstance(prefix, str)
        assert isinstance(middle, str)
        assert isinstance(suffix, str)
        span_length = len(middle.split())
        return {
            "kind": "infilling",
            "prefix": prefix,
            "middle": middle,
            "suffix": suffix,
            "domain": str(parameters.get("domain", row.get("domain", manifest.id))),
            "span_length": span_length,
            "prefix_length": len(prefix.split()),
            "suffix_length": len(suffix.split()),
            "span_length_bin": str(
                row.get("span_length_bin", _span_length_bin(span_length))
            ),
            "boundary_density": str(
                row.get("boundary_density", _boundary_density(prefix, suffix))
            ),
            "metadata": metadata,
        }
    if strategy == "editing":
        source = str(_mapped(row, mapping, "source_text"))
        reference = str(_mapped(row, mapping, "reference_edit"))
        raw_targets = _mapped(row, mapping, "target_spans", ())
        raw_non_targets = _mapped(row, mapping, "non_target_spans", ())
        if raw_targets or raw_non_targets:
            targets = tuple(TextSpan.model_validate(value) for value in raw_targets)
            non_targets = tuple(
                TextSpan.model_validate(value) for value in raw_non_targets
            )
        else:
            targets, non_targets = _derive_edit_spans(source, reference)
        return {
            "kind": "editing",
            "source_text": source,
            "instruction": str(_mapped(row, mapping, "instruction")),
            "reference_edit": reference,
            "target_spans": targets,
            "non_target_spans": non_targets,
            "minimum_edit_distance": int(
                _mapped(
                    row,
                    mapping,
                    "minimum_edit_distance",
                    _levenshtein(source, reference),
                )
            ),
            "edit_type": str(_mapped(row, mapping, "edit_type", manifest.id)),
            "evidence": _strings(_mapped(row, mapping, "evidence", ())),
            "metadata": metadata,
        }
    if strategy == "reasoning":
        choice_fields = parameters.get("choice_fields")
        if isinstance(choice_fields, list):
            choices = tuple(str(_field(row, str(field))) for field in choice_fields)
        else:
            choices_value = _mapped(row, mapping, "choices", ())
            choices = _strings(choices_value) if choices_value else ()
        answer_index = _mapped(row, mapping, "answer_index")
        if answer_index is not None:
            answer_index = int(answer_index) - _int_parameter(
                parameters, "answer_index_offset", 0
            )
        answer = _mapped(row, mapping, "answer")
        if (
            choices
            and answer_index is not None
            and (answer is None or str(answer).strip().isdigit())
        ):
            try:
                answer = choices[answer_index]
            except IndexError as exc:
                raise DatasetError(
                    f"reasoning answer index {answer_index} is outside "
                    f"{len(choices)} choices"
                ) from exc
        return {
            "kind": "reasoning",
            "problem": str(_mapped(row, mapping, "problem")),
            "answer": str(answer),
            "reasoning_type": str(parameters["reasoning_type"]),
            "choices": choices,
            "answer_index": answer_index,
            "metadata": metadata,
        }
    raise DatasetError(f"unsupported preprocessing strategy: {strategy}")


def preprocess_rows(
    manifest: DatasetManifest,
    rows: Iterable[Mapping[str, Any]],
    split: str,
) -> NormalizedDataset:
    """Normalize raw rows and assign stable content-addressed IDs."""
    records: list[SampleRecord] = []
    for index, row in enumerate(rows):
        parameters = manifest.preprocessing.parameters
        if (
            row.get("kind") is None
            and manifest.preprocessing.strategy == "infilling"
            and bool(parameters.get("skip_short", False))
        ):
            text = _mapped(row, manifest.preprocessing.field_map, "text", "")
            minimum_words = _int_parameter(parameters, "min_words", 9)
            if len(str(text).split()) < minimum_words:
                continue
        source_id = _source_id(row, index)
        payload = _normalize_payload(manifest, row)
        provisional = parse_sample_record(
            _with_identity(manifest, split, source_id, payload)
        )
        normalized_payload = provisional.model_dump(mode="json", exclude_none=False)
        normalized_payload["sample_id"] = expected_sample_id(provisional)
        record = parse_sample_record(normalized_payload)
        if record.kind != manifest.record_kind:
            raise DatasetError(
                f"manifest {manifest.id} expects {manifest.record_kind}, "
                f"preprocessor produced {record.kind}"
            )
        records.append(record)
    return NormalizedDataset(manifest.id, split, records)


def prepare_suite(
    registry_path: str | Path,
    suite: str,
    output_root: str | Path,
    *,
    mode: str = "sample",
    cache_root: str | Path | None = None,
    seed: int = 0,
    shuffle: bool = False,
    sample_limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare every split in a registered suite and write a suite manifest."""
    if mode not in {"sample", "full"}:
        raise DatasetError("prepare mode must be 'sample' or 'full'")
    registry_file = Path(registry_path).resolve()
    data_root = registry_file.parent
    registry = load_data_registry(registry_file)
    try:
        suite_spec = registry.suites[suite]
    except KeyError as exc:
        available = ", ".join(sorted(registry.suites))
        raise DatasetError(f"unknown suite {suite!r}; available: {available}") from exc
    destination_root = Path(output_root).resolve() / suite
    cache = Path(cache_root).resolve() if cache_root else data_root / "cache"
    dataset_entries: list[dict[str, JsonValue]] = []
    for dataset_id in suite_spec.datasets:
        manifest_path = data_root / registry.datasets[dataset_id]
        manifest = load_dataset_manifest(manifest_path)
        if manifest.id != dataset_id:
            raise DatasetError(
                f"registry key {dataset_id!r} does not match "
                f"manifest id {manifest.id!r}"
            )
        for split in manifest.splits:
            if mode == "sample":
                rows = load_bundled_sample_rows(manifest, data_root, split)
            else:
                raw_path = acquire_full_source(
                    manifest,
                    split,
                    cache,
                    force=force,
                )
                rows = tuple(read_jsonl_rows(raw_path))
            normalized = preprocess_rows(manifest, rows, split)
            frozen, lock = freeze_split(
                normalized,
                manifest.manifest_hash,
                seed=seed,
                shuffle=shuffle,
                sample_limit=sample_limit,
            )
            dataset_directory = destination_root / manifest.id
            dataset_path = dataset_directory / f"{split}.jsonl"
            lock_path = dataset_directory / f"{split}.split-lock.json"
            frozen.write_jsonl(dataset_path)
            write_split_lock(lock_path, lock)
            dataset_entries.append(
                {
                    "dataset_id": manifest.id,
                    "split": split,
                    "record_kind": manifest.record_kind,
                    "sample_count": len(frozen),
                    "dataset_hash": frozen.dataset_hash,
                    "manifest_hash": manifest.manifest_hash,
                    "lock_hash": lock.lock_hash,
                    "data_path": str(dataset_path.relative_to(destination_root)),
                    "lock_path": str(lock_path.relative_to(destination_root)),
                }
            )
    suite_manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "suite": suite,
        "mode": mode,
        "seed": seed,
        "shuffle": shuffle,
        "sample_limit": sample_limit,
        "datasets": dataset_entries,
    }
    from dimebench.artifacts.hashing import hash_json

    suite_manifest["suite_hash"] = hash_json(dataset_entries)
    write_json_atomic(destination_root / "suite-manifest.json", suite_manifest)
    return suite_manifest


__all__ = ["prepare_suite", "preprocess_rows"]

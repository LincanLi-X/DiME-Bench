from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.utils.config import resolve_path
from src.utils.io import ensure_dir, read_json, write_json


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s| is)\s+(?:the\s+)?(?:edited|corrected)\s+text\s*:|edited\s+text\s*:|corrected\s+sentence\s*:)\s*",
    re.IGNORECASE,
)


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")


def normalize_for_match(text: str) -> str:
    text = (text or "").strip()
    text = PREAMBLE_RE.sub("", text).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_edited_text(model_output: str) -> str:
    text = normalize_for_match(model_output)
    if not text:
        return ""
    markers = [
        "\n\nExplanation:",
        "\nExplanation:",
        "\n\nChanges:",
        "\nChanges:",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].lower().rstrip(":") in {"edited text", "corrected sentence", "output"}:
        return normalize_for_match("\n".join(lines[1:]))
    return normalize_for_match(text)


def edit_distance(a: list[str], b: list[str]) -> int:
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            distance += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            distance += i2 - i1
        elif tag == "insert":
            distance += j2 - j1
    return distance


def diff_target_token_indices(source: str, reference: str) -> list[int]:
    source_tokens = tokenize_text(source)
    reference_tokens = tokenize_text(reference)
    matcher = SequenceMatcher(a=source_tokens, b=reference_tokens, autojunk=False)
    indices: set[int] = set()
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            indices.update(range(i1, i2))
            if i1 == i2 and source_tokens:
                indices.add(min(i1, len(source_tokens) - 1))
    return sorted(indices)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_text(normalize_for_match(prediction).lower())
    ref_tokens = tokenize_text(normalize_for_match(reference).lower())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    used = [False] * len(ref_tokens)
    overlap = 0
    for token in pred_tokens:
        for idx, ref_token in enumerate(ref_tokens):
            if not used[idx] and token == ref_token:
                used[idx] = True
                overlap += 1
                break
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def preservation_score(source: str, output: str, target_indices: list[int] | None = None) -> float:
    source_tokens = tokenize_text(source)
    output_tokens = tokenize_text(output)
    if not source_tokens:
        return 1.0
    target_set = set(target_indices or [])
    non_target = [tok.lower() for idx, tok in enumerate(source_tokens) if idx not in target_set]
    if not non_target:
        return 1.0
    output_lower = [tok.lower() for tok in output_tokens]
    matcher = SequenceMatcher(a=non_target, b=output_lower, autojunk=False)
    kept = sum(block.size for block in matcher.get_matching_blocks())
    return kept / len(non_target)


def contradiction_count(sample: dict[str, Any], text: str) -> int:
    normalized = normalize_for_match(text).lower()
    forbidden = [str(x).lower() for x in sample.get("forbidden_phrases", []) if str(x).strip()]
    if forbidden:
        return sum(1 for phrase in forbidden if phrase in normalized)
    before = sample.get("contradiction_before")
    if isinstance(before, list):
        return sum(1 for phrase in before if str(phrase).lower() in normalized)
    if isinstance(before, str) and before.lower() in normalized:
        return 1
    return 0


def _contains_required(sample: dict[str, Any], text: str) -> bool:
    normalized = normalize_for_match(text).lower()
    required = [str(x).lower() for x in sample.get("required_phrases", []) if str(x).strip()]
    if not required:
        return True
    return all(phrase in normalized for phrase in required)


def score_track3_sample(sample: dict[str, Any], parsed_output: str, success_f1_threshold: float = 0.90) -> dict[str, Any]:
    source = sample.get("source_text", "")
    reference = sample.get("reference", "")
    dataset = sample.get("dataset")
    source_tokens = tokenize_text(source)
    output_tokens = tokenize_text(parsed_output)
    reference_tokens = tokenize_text(reference)
    target_indices = sample.get("target_token_indices")
    if target_indices is None:
        target_indices = diff_target_token_indices(source, reference)

    changed_tokens = edit_distance(source_tokens, output_tokens)
    minimal_edit_distance = int(
        sample.get("minimal_edit_distance")
        if sample.get("minimal_edit_distance") is not None
        else edit_distance(source_tokens, reference_tokens)
    )
    minimal_edit_distance = max(1, minimal_edit_distance)
    changed_token_ratio = changed_tokens / max(1, len(source_tokens))
    over_edit = changed_tokens > float(sample.get("over_edit_multiplier", 1.5)) * minimal_edit_distance
    ref_f1 = token_f1(parsed_output, reference)
    normalized_pred = normalize_for_match(parsed_output).lower()
    normalized_ref = normalize_for_match(reference).lower()

    if dataset == "synthetic_contradiction_repair":
        edit_success = contradiction_count(sample, parsed_output) == 0 and _contains_required(sample, parsed_output)
    else:
        edit_success = normalized_pred == normalized_ref or ref_f1 >= success_f1_threshold

    before_count = int(sample.get("contradiction_count_before", 0) or 0)
    after_count = contradiction_count(sample, parsed_output)
    contradiction_reduction = None
    if before_count > 0:
        contradiction_reduction = max(0.0, min(1.0, (before_count - after_count) / before_count))

    return {
        "prediction": parsed_output,
        "gold": reference,
        "edit_success": bool(edit_success),
        "reference_token_f1": ref_f1,
        "preservation_score": preservation_score(source, parsed_output, target_indices),
        "over_edit": bool(over_edit),
        "changed_tokens": changed_tokens,
        "minimal_edit_distance": minimal_edit_distance,
        "changed_token_ratio": changed_token_ratio,
        "contradiction_count_before": before_count,
        "contradiction_count_after": after_count,
        "contradiction_reduction": contradiction_reduction,
        "parser_failed": not bool(parsed_output.strip()),
    }


def summarize_track3_rows(rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    scores = [row.get("score") or {} for row in rows]
    n = len(scores)
    if n == 0:
        return {
            "track": "track3_editing",
            "dataset": dataset_name,
            "num_samples": 0,
            "metric_name": "edit_success",
            "metric_value": 0.0,
            "parser_failure_rate": 0.0,
            "result_status": "empty",
        }

    def mean(key: str) -> float | None:
        values = [score.get(key) for score in scores if score.get(key) is not None]
        if not values:
            return None
        return sum(float(value) for value in values) / len(values)

    edit_success = mean("edit_success") or 0.0
    parser_failure_rate = mean("parser_failed") or 0.0
    return {
        "track": "track3_editing",
        "dataset": dataset_name,
        "num_samples": n,
        "metric_name": "edit_success",
        "metric_value": edit_success,
        "edit_success": edit_success,
        "preservation_score": mean("preservation_score"),
        "over_edit_rate": mean("over_edit"),
        "contradiction_reduction": mean("contradiction_reduction"),
        "changed_token_ratio": mean("changed_token_ratio"),
        "parser_failure_rate": parser_failure_rate,
        "num_errors": sum(1 for row in rows if row.get("error")),
        "result_status": "ok",
    }


def write_track3_analysis_tables(metric_root: str | Path = "metrics/track3_editing") -> dict[str, str]:
    root = ensure_dir(resolve_path(metric_root))
    points: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    for metric_path in sorted(root.glob("*/*.json")):
        metric = read_json(metric_path)
        if metric.get("track") != "track3_editing":
            continue
        points.append(
            {
                "model": metric.get("model"),
                "dataset": metric.get("dataset"),
                "edit_success": metric.get("edit_success"),
                "preservation_score": metric.get("preservation_score"),
                "over_edit_rate": metric.get("over_edit_rate"),
                "contradiction_reduction": metric.get("contradiction_reduction"),
                "num_samples": metric.get("num_samples"),
                "result_status": metric.get("result_status"),
            }
        )
        distribution.append(
            {
                "model": metric.get("model"),
                "dataset": metric.get("dataset"),
                "changed_token_ratio": metric.get("changed_token_ratio"),
                "over_edit_rate": metric.get("over_edit_rate"),
                "num_samples": metric.get("num_samples"),
            }
        )

    points_path = root / "preservation_correction_points.csv"
    dist_path = root / "over_edit_distribution.csv"
    _write_csv(points_path, points)
    _write_csv(dist_path, distribution)
    write_json(root / "preservation_correction_points.json", points)
    write_json(root / "over_edit_distribution.json", distribution)
    return {
        "preservation_correction_points": str(points_path),
        "over_edit_distribution": str(dist_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sample = {
        "dataset": "synthetic_contradiction_repair",
        "source_text": "Maya moved to Paris in 2018. Later, Maya had never lived in Paris.",
        "reference": "Maya moved to Paris in 2018. Later, Maya lived in Paris.",
        "required_phrases": ["lived in Paris"],
        "forbidden_phrases": ["never lived in Paris"],
        "contradiction_count_before": 1,
    }
    parsed = parse_edited_text("Here is the edited text: Maya moved to Paris in 2018. Later, Maya lived in Paris.")
    score = score_track3_sample(sample, parsed)
    print(score)
    if not score["edit_success"]:
        raise SystemExit("Track 3 metric sanity check failed")


if __name__ == "__main__":
    main()

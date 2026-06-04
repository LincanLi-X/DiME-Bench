from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from src.parsing.answer_extractors import extract_choice


def parse_track4_output(dataset_name: str, output: str, sample: dict[str, Any]) -> Any:
    if dataset_name == "gsm8k":
        return extract_math_answer(output, numeric_only=True)
    if dataset_name == "math_500":
        return extract_math_answer(output, numeric_only=False)
    if dataset_name == "winogrande":
        return extract_choice(output, ["A", "B"])
    if dataset_name == "path_star":
        return extract_path(output, sample)
    raise ValueError(f"Unsupported Track 4 dataset: {dataset_name}")


def score_track4_sample(sample: dict[str, Any], dataset_name: str, output: str) -> dict[str, Any]:
    parsed = parse_track4_output(dataset_name, output, sample)
    reference = sample.get("reference", sample.get("answer"))
    parser_failed = parsed is None or parsed == []

    if dataset_name == "gsm8k":
        is_correct = exact_math_match(parsed, reference, numeric=True)
        return {
            "prediction": parsed,
            "gold": reference,
            "is_correct": is_correct,
            "parser_failed": parser_failed,
        }
    if dataset_name == "math_500":
        is_correct = exact_math_match(parsed, reference, numeric=False)
        return {
            "prediction": parsed,
            "gold": reference,
            "is_correct": is_correct,
            "parser_failed": parser_failed,
        }
    if dataset_name == "winogrande":
        is_correct = parsed == reference
        return {
            "prediction": parsed,
            "gold": reference,
            "is_correct": is_correct,
            "parser_failed": parser_failed,
        }
    if dataset_name == "path_star":
        validity = path_validity(parsed or [], sample)
        is_correct = bool(validity["valid_path"] and validity["reaches_target"] and validity["exact_match"])
        return {
            "prediction": parsed,
            "gold": reference,
            "is_correct": is_correct,
            "parser_failed": parser_failed,
            **validity,
        }
    raise ValueError(f"Unsupported Track 4 dataset: {dataset_name}")


def summarize_track4_rows(rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    num_samples = len(rows)
    scores = [(row.get("score") or {}) for row in rows]
    correct = sum(1 for score in scores if score.get("is_correct"))
    parser_failures = sum(1 for score in scores if score.get("parser_failed"))
    num_errors = sum(1 for row in rows if row.get("error"))
    metric_name = {
        "gsm8k": "exact_match",
        "math_500": "exact_match",
        "winogrande": "accuracy",
        "path_star": "path_validity",
    }[dataset_name]
    metric_value = correct / num_samples if num_samples else 0.0
    parser_failure_rate = parser_failures / num_samples if num_samples else 0.0
    result_status = "ok" if num_errors == 0 else "partial_error"
    if num_samples == 0:
        result_status = "empty"
    return {
        "metric_name": metric_name,
        "metric_value": metric_value,
        metric_name: metric_value,
        "parser_failure_rate": parser_failure_rate,
        "num_errors": num_errors,
        "result_status": result_status,
    }


def aggregate_by_reasoning_type(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {"chaining": [], "planning": []}
    for row in metric_rows:
        reasoning_type = row.get("reasoning_type")
        value = row.get("metric_value")
        if reasoning_type in grouped and isinstance(value, (int, float)):
            grouped[reasoning_type].append(float(value))
    chain_avg = _mean(grouped["chaining"])
    planning_avg = _mean(grouped["planning"])
    return {
        "chain_average": chain_avg,
        "planning_average": planning_avg,
        "planning_vs_chaining_gap": None
        if chain_avg is None or planning_avg is None
        else planning_avg - chain_avg,
    }


def extract_math_answer(output: str, numeric_only: bool) -> str | None:
    text = str(output).strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalize_math_text(boxed[-1], numeric_only=numeric_only)
    final = re.findall(
        r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )
    if final:
        return normalize_math_text(final[-1], numeric_only=numeric_only)
    if numeric_only:
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?", text)
        if nums:
            return normalize_number(nums[-1])
        return None
    tail = text.splitlines()[-1] if text.splitlines() else text
    return normalize_math_text(tail, numeric_only=False)


def exact_math_match(prediction: Any, reference: Any, numeric: bool) -> bool:
    if prediction is None:
        return False
    if numeric:
        return normalize_number(str(prediction)) == normalize_number(str(reference))
    return normalize_math_text(str(prediction), numeric_only=False) == normalize_math_text(
        str(reference), numeric_only=False
    )


def normalize_math_text(text: str, numeric_only: bool = False) -> str:
    text = str(text).strip()
    text = re.sub(r"^[$\\boxed\{\}\s]+|[$\\boxed\{\}\s]+$", "", text)
    text = text.replace(",", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    if numeric_only:
        return normalize_number(text)
    return text.lower()


def normalize_number(text: str) -> str:
    text = str(text).strip().replace(",", "")
    text = re.sub(r"[^0-9+\-./]", "", text)
    if "/" in text and text.count("/") == 1:
        num, den = text.split("/")
        try:
            value = Decimal(num) / Decimal(den)
            return format(value.normalize(), "f").rstrip("0").rstrip(".") or "0"
        except (InvalidOperation, ZeroDivisionError):
            return text
    try:
        value = Decimal(text)
        return format(value.normalize(), "f").rstrip("0").rstrip(".") or "0"
    except InvalidOperation:
        return text


def extract_path(output: str, sample: dict[str, Any]) -> list[str]:
    known_nodes = {str(node) for arm in sample.get("arms", []) for node in arm}
    text = str(output).strip()
    if "Path:" in text:
        text = text.rsplit("Path:", 1)[-1]
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    tokens = re.findall(r"[A-Za-z0-9_:-]+", first_line.replace("->", " "))
    if not tokens:
        tokens = re.findall(r"[A-Za-z0-9_:-]+", text.replace("->", " "))
    if known_nodes:
        filtered = [token for token in tokens if token in known_nodes]
        return filtered
    return tokens


def path_validity(path: list[str], sample: dict[str, Any]) -> dict[str, Any]:
    reference = [str(node) for node in sample.get("reference", sample.get("answer", []))]
    edge_set = set()
    for arm in sample.get("arms", []):
        arm = [str(node) for node in arm]
        for left, right in zip(arm, arm[1:]):
            edge_set.add((left, right))
    starts_correct = bool(path) and path[0] == str(sample.get("start"))
    reaches_target = bool(path) and path[-1] == str(sample.get("target"))
    valid_edges = all((left, right) in edge_set for left, right in zip(path, path[1:]))
    valid_path = bool(path) and starts_correct and valid_edges
    return {
        "valid_path": valid_path,
        "starts_correct": starts_correct,
        "reaches_target": reaches_target,
        "valid_edges": valid_edges,
        "exact_match": path == reference,
        "path_length": len(path),
        "reference_length": len(reference),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)

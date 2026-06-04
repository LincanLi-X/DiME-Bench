from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any


NA = "N/A"


def parse_json_object(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def score_track6_output(sample: dict[str, Any], dataset_name: str, output: str) -> dict[str, Any]:
    if dataset_name == "json_schema_generation":
        return _score_json_schema(sample, output)
    if dataset_name == "synthetic_contradiction_repair":
        return _score_repair(sample, output)
    if dataset_name == "fixed_length":
        return _score_fixed_length(sample, output)
    return _score_infilling(sample, output)


def _score_infilling(sample: dict[str, Any], output: str) -> dict[str, Any]:
    reference = str(sample.get("reference") or "")
    f1 = token_f1(output, reference)
    return {
        "metric_name": "token_f1",
        "metric_value": f1,
        "quality_score": f1,
        "parser_failed": False,
        "prediction": output.strip(),
        "gold": reference,
    }


def _score_repair(sample: dict[str, Any], output: str) -> dict[str, Any]:
    reference = str(sample.get("reference") or "")
    f1 = token_f1(output, reference)
    markers = sample.get("contradiction_markers") or ["contradiction", "inconsistent", "however"]
    lower = output.lower()
    cleanup = 1.0 - min(1.0, sum(1 for marker in markers if str(marker).lower() in lower) / max(1, len(markers)))
    score = 0.8 * f1 + 0.2 * cleanup
    return {
        "metric_name": "repair_score",
        "metric_value": score,
        "quality_score": score,
        "token_f1": f1,
        "cleanup_score": cleanup,
        "parser_failed": False,
        "prediction": output.strip(),
        "gold": reference,
    }


def _score_json_schema(sample: dict[str, Any], output: str) -> dict[str, Any]:
    parsed = parse_json_object(output)
    required = _required_keys(sample)
    if not isinstance(parsed, dict):
        return {
            "metric_name": "json_schema_score",
            "metric_value": 0.0,
            "quality_score": 0.0,
            "json_valid": False,
            "required_key_coverage": 0.0,
            "parser_failed": True,
            "prediction": None,
            "gold": sample.get("reference"),
        }
    coverage = sum(1 for key in required if key in parsed) / max(1, len(required))
    score = 0.5 + 0.5 * coverage
    return {
        "metric_name": "json_schema_score",
        "metric_value": score,
        "quality_score": score,
        "json_valid": True,
        "required_key_coverage": coverage,
        "parser_failed": False,
        "prediction": parsed,
        "gold": sample.get("reference"),
    }


def _score_fixed_length(sample: dict[str, Any], output: str) -> dict[str, Any]:
    target = int(sample.get("target_output_length") or sample.get("max_new_tokens") or 1)
    count = len(_tokens(output))
    adherence = max(0.0, 1.0 - abs(count - target) / max(1, target))
    return {
        "metric_name": "length_adherence",
        "metric_value": adherence,
        "quality_score": adherence,
        "token_count": count,
        "target_output_length": target,
        "parser_failed": False,
        "prediction": output.strip(),
        "gold": sample.get("reference"),
    }


def token_f1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    gold = _tokens(reference)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pred_counts = Counter(pred)
    gold_counts = Counter(gold)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def summarize_output_rows(rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    scores = [float((row.get("score") or {}).get("quality_score", 0.0)) for row in rows if not row.get("error")]
    parser_failures = [
        bool((row.get("score") or {}).get("parser_failed")) or bool(row.get("error")) for row in rows
    ]
    metric_name = rows[0].get("score", {}).get("metric_name") if rows else _default_metric_name(dataset_name)
    metric_value = sum(scores) / len(scores) if scores else 0.0
    return {
        "metric_name": metric_name or _default_metric_name(dataset_name),
        "metric_value": metric_value,
        "parser_failure_rate": sum(parser_failures) / len(parser_failures) if parser_failures else 0.0,
        "num_errors": sum(1 for row in rows if row.get("error")),
    }


def compute_trajectory_metrics(
    trajectory_rows: list[dict[str, Any]],
    model_family: str,
) -> dict[str, Any]:
    if model_family != "dllm":
        return {"str": NA, "mean_trc": NA, "foe": NA, "trajectory_available": False}
    full_rows = [row for row in trajectory_rows if row.get("token_ids") is not None]
    if not full_rows:
        return {
            "str": NA,
            "mean_trc": NA,
            "foe": NA,
            "trajectory_available": False,
            "partial_trajectory_available": bool(trajectory_rows),
        }
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_sample[str(row["sample_id"])].append(row)

    str_values = []
    trc_values = []
    foe_values = []
    for sample_rows in by_sample.values():
        sample_rows = sorted(sample_rows, key=lambda row: int(row.get("step", 0)))
        final_tokens = sample_rows[-1].get("token_ids") or []
        finalized_step: dict[int, int] = {}
        stable_new = 0
        total_new = 0
        values_by_pos: dict[int, set[int]] = defaultdict(set)
        previous_unmasked: set[int] = set()
        for row in sample_rows:
            step = int(row.get("step", 0))
            token_ids = row.get("token_ids") or []
            mask_status = row.get("mask_status") or []
            unmasked = {idx for idx, is_mask in enumerate(mask_status) if not is_mask}
            newly_unmasked = unmasked - previous_unmasked
            for idx in newly_unmasked:
                finalized_step.setdefault(idx, step)
                total_new += 1
                if idx < len(token_ids) and idx < len(final_tokens) and token_ids[idx] == final_tokens[idx]:
                    stable_new += 1
            for idx, tok in enumerate(token_ids):
                if idx < len(mask_status) and not mask_status[idx]:
                    values_by_pos[idx].add(int(tok))
            previous_unmasked = unmasked
        if total_new:
            str_values.append(stable_new / total_new)
        for values in values_by_pos.values():
            trc_values.append(len(values))
        if finalized_step:
            foe_values.append(_finalization_order_entropy(finalized_step, len(final_tokens)))
    return {
        "str": _mean(str_values) if str_values else NA,
        "mean_trc": _mean(trc_values) if trc_values else NA,
        "foe": _mean(foe_values) if foe_values else NA,
        "trajectory_available": True,
        "partial_trajectory_available": True,
    }


def _finalization_order_entropy(finalized_step: dict[int, int], length: int, bins: int = 8) -> float:
    if length <= 0:
        return 0.0
    counts = [0] * bins
    for pos in finalized_step:
        bin_idx = min(bins - 1, int(pos / length * bins))
        counts[bin_idx] += 1
    total = sum(counts)
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return entropy / math.log(bins)


def _required_keys(sample: dict[str, Any]) -> list[str]:
    schema = sample.get("schema") or (sample.get("prompt_fields") or {}).get("schema")
    if isinstance(schema, str):
        parsed = parse_json_object(schema)
    else:
        parsed = schema
    if isinstance(parsed, dict):
        required = parsed.get("required")
        if isinstance(required, list):
            return [str(key) for key in required]
        properties = parsed.get("properties")
        if isinstance(properties, dict):
            return [str(key) for key in properties.keys()]
    reference = sample.get("reference")
    if isinstance(reference, dict):
        return [str(key) for key in reference.keys()]
    return []


def _default_metric_name(dataset_name: str) -> str:
    return {
        "cnn_dailymail_infilling": "token_f1",
        "synthetic_contradiction_repair": "repair_score",
        "json_schema_generation": "json_schema_score",
        "fixed_length": "length_adherence",
    }.get(dataset_name, "quality_score")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text).lower())


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

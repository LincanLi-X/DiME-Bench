from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.utils.io import ensure_dir, write_json


TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b", re.UNICODE)
NEGATIONS = {"no", "not", "never", "none", "neither", "without", "cannot", "can't", "won't"}
AFFIRMATIONS = {"is", "are", "was", "were", "can", "will", "has", "have", "does", "do"}


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = defaultdict(int)
    for token in ref_tokens:
        ref_counts[token] += 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts[token] > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    beta = precision / (recall + 1e-12)
    return ((1 + beta * beta) * precision * recall) / (recall + beta * beta * precision + 1e-12)


def _lcs_len(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def boundary_consistency(prediction: str, prefix: str, suffix: str) -> float:
    pred = normalize_text(prediction)
    if not pred:
        return 0.0
    if _label_leak(pred):
        return 0.0

    prefix_tail = tokenize(prefix)[-16:]
    suffix_head = tokenize(suffix)[:16]
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0

    copied_prefix = _longest_common_run(prefix_tail, pred_tokens[:32])
    copied_suffix = _longest_common_run(suffix_head, pred_tokens[-32:])
    if copied_prefix >= 8 or copied_suffix >= 8:
        return 0.0

    score = 1.0
    if prefix_tail and pred_tokens[0] == prefix_tail[-1]:
        score -= 0.25
    if suffix_head and pred_tokens[-1] == suffix_head[0]:
        score -= 0.25
    if len(pred_tokens) < 3:
        score -= 0.25
    return max(0.0, score)


def contradiction_flag(prediction: str, prefix: str, suffix: str) -> bool:
    """Deterministic placeholder until an NLI or LLM-judge evaluator is wired in."""
    pred_tokens = set(tokenize(prediction))
    boundary_tokens = set(tokenize(prefix[-1000:] + " " + suffix[:1000]))
    if not pred_tokens:
        return True
    pred_neg = bool(pred_tokens & NEGATIONS)
    boundary_neg = bool(boundary_tokens & NEGATIONS)
    pred_aff = bool(pred_tokens & AFFIRMATIONS)
    boundary_aff = bool(boundary_tokens & AFFIRMATIONS)
    return (pred_neg and boundary_aff and not boundary_neg) or (boundary_neg and pred_aff and not pred_neg)


def format_failure(prediction: str, prefix: str, suffix: str) -> bool:
    pred = normalize_text(prediction)
    if not pred:
        return True
    if _label_leak(pred):
        return True
    prefix_norm = normalize_text(prefix)
    suffix_norm = normalize_text(suffix)
    if len(pred) > 80 and (pred in prefix_norm or pred in suffix_norm):
        return True
    pred_tokens = tokenize(prediction)
    if len(pred_tokens) > 20:
        prefix_overlap = token_f1(prediction, " ".join(tokenize(prefix)[-len(pred_tokens) :]))
        suffix_overlap = token_f1(prediction, " ".join(tokenize(suffix)[: len(pred_tokens)]))
        if max(prefix_overlap, suffix_overlap) > 0.85:
            return True
    return False


def score_infilling_sample(sample: dict[str, Any], prediction: str) -> dict[str, Any]:
    reference = sample["reference"]
    prefix = sample.get("prefix", "")
    suffix = sample.get("suffix", "")
    failure = format_failure(prediction, prefix, suffix)
    return {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge_l": rouge_l(prediction, reference),
        "bertscore_f1": None,
        "boundary_consistency": boundary_consistency(prediction, prefix, suffix),
        "contradiction": contradiction_flag(prediction, prefix, suffix),
        "format_failure": failure,
        "prediction_length": len(tokenize(prediction)),
        "reference_length": len(tokenize(reference)),
    }


def add_bertscore(rows: list[dict[str, Any]], lang: str = "en") -> dict[str, Any]:
    try:
        from bert_score import score as bert_score
    except Exception as exc:
        return {
            "available": False,
            "note": f"BERTScore unavailable; install bert_score to enable it. Import error: {exc!r}",
        }
    predictions = [row.get("parsed_output") or "" for row in rows]
    references = [row.get("reference") or "" for row in rows]
    if not predictions:
        return {"available": True, "note": "No rows to score."}
    _, _, f1 = bert_score(predictions, references, lang=lang, verbose=False)
    values = [float(x) for x in f1.tolist()]
    for row, value in zip(rows, values):
        row.setdefault("score", {})["bertscore_f1"] = value
    return {"available": True, "model": "bert_score default", "num_scored": len(values)}


def summarize_infilling_rows(
    rows: list[dict[str, Any]],
    dataset_name: str,
    model_name: str,
    runtime_s: float,
    track: str = "track2_infilling",
) -> dict[str, Any]:
    scores = [row.get("score") or {} for row in rows]
    num_samples = len(rows)
    parser_failures = sum(1 for score in scores if score.get("format_failure"))
    errors = sum(1 for row in rows if row.get("error"))

    metric_values = {
        "exact_match": _mean_score(scores, "exact_match"),
        "token_f1": _mean_score(scores, "token_f1"),
        "rouge_l": _mean_score(scores, "rouge_l"),
        "bertscore_f1": _mean_score(scores, "bertscore_f1"),
        "boundary_consistency": _mean_score(scores, "boundary_consistency"),
        "contradiction_rate": _mean_bool(scores, "contradiction"),
    }
    primary = "token_f1" if dataset_name == "wikitext103_sentence" else "rouge_l"
    return {
        "track": track,
        "dataset": dataset_name,
        "model": model_name,
        "num_samples": num_samples,
        "metric_name": primary,
        "metric_value": metric_values[primary],
        "parser_failure_rate": parser_failures / num_samples if num_samples else 0.0,
        "runtime_s": runtime_s,
        "result_status": "ok" if errors == 0 else "completed_with_errors",
        "num_errors": errors,
        "metrics": metric_values,
        "contradiction_evaluator": "heuristic_negation_placeholder",
    }


def write_breakdowns(rows: list[dict[str, Any]], out_prefix: Path) -> dict[str, str]:
    ensure_dir(out_prefix.parent)
    span_rows = _breakdown(rows, "span_length_bin")
    density_rows = _breakdown(rows, "context_density_bin")
    span_json = out_prefix.with_name(out_prefix.name + "_by_span_length.json")
    span_csv = out_prefix.with_name(out_prefix.name + "_by_span_length.csv")
    density_json = out_prefix.with_name(out_prefix.name + "_by_context_density.json")
    density_csv = out_prefix.with_name(out_prefix.name + "_by_context_density.csv")
    write_json(span_json, span_rows)
    write_json(density_json, density_rows)
    _write_csv(span_csv, span_rows)
    _write_csv(density_csv, density_rows)
    return {
        "by_span_length_json": str(span_json),
        "by_span_length_csv": str(span_csv),
        "by_context_density_json": str(density_json),
        "by_context_density_csv": str(density_csv),
    }


def _breakdown(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = row.get("sample") or {}
        grouped[str(sample.get(key) or "unknown")].append(row)
    out = []
    for bucket, bucket_rows in sorted(grouped.items()):
        scores = [row.get("score") or {} for row in bucket_rows]
        out.append(
            {
                key: bucket,
                "num_samples": len(bucket_rows),
                "exact_match": _mean_score(scores, "exact_match"),
                "token_f1": _mean_score(scores, "token_f1"),
                "rouge_l": _mean_score(scores, "rouge_l"),
                "bertscore_f1": _mean_score(scores, "bertscore_f1"),
                "boundary_consistency": _mean_score(scores, "boundary_consistency"),
                "contradiction_rate": _mean_bool(scores, "contradiction"),
                "parser_failure_rate": _mean_bool(scores, "format_failure"),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "span_length_bin",
        "context_density_bin",
        "num_samples",
        "exact_match",
        "token_f1",
        "rouge_l",
        "bertscore_f1",
        "boundary_consistency",
        "contradiction_rate",
        "parser_failure_rate",
    ]
    present = [name for name in fieldnames if any(name in row for row in rows)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=present)
        writer.writeheader()
        writer.writerows(rows)


def _mean_score(scores: list[dict[str, Any]], key: str) -> float | None:
    values = [score.get(key) for score in scores if score.get(key) is not None]
    if not values:
        return None
    return float(mean(float(value) for value in values))


def _mean_bool(scores: list[dict[str, Any]], key: str) -> float:
    if not scores:
        return 0.0
    return sum(1 for score in scores if bool(score.get(key))) / len(scores)


def _label_leak(text: str) -> bool:
    lowered = text.lower()
    labels = ["prefix:", "suffix:", "missing middle", "masked middle", "return only"]
    return any(label in lowered for label in labels)


def _longest_common_run(a: list[str], b: list[str]) -> int:
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            run = 0
            while i + run < len(a) and j + run < len(b) and a[i + run] == b[j + run]:
                run += 1
            best = max(best, run)
    return best

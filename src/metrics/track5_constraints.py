from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from src.metrics.accuracy import accuracy
from src.metrics.ifeval_simple import evaluate_ifeval_fallback


def score_track5_sample(sample: dict[str, Any], dataset_name: str, output: str) -> dict[str, Any]:
    if dataset_name == "mt_bench":
        return score_mt_bench(sample, output)
    if dataset_name == "ifeval":
        return score_ifeval(sample, output)
    if dataset_name == "json_schema":
        return score_json_schema(sample, output)
    if dataset_name == "spider":
        return score_spider(sample, output)
    raise ValueError(f"Unsupported Track 5 dataset: {dataset_name}")


def score_mt_bench(sample: dict[str, Any], output: str) -> dict[str, Any]:
    return {
        "prediction": output,
        "is_correct": None,
        "judge_score": None,
        "length": len(output.split()),
        "result_status": "pending_judge",
        "warning": "MT-Bench requires an external judge; candidate output was captured only.",
    }


def score_ifeval(sample: dict[str, Any], output: str) -> dict[str, Any]:
    result = evaluate_ifeval_fallback(sample, output)
    return {
        "prediction": output,
        "gold": sample.get("instruction_id_list"),
        "is_correct": bool(result["strict_pass"]),
        "ifeval_fallback": result,
        "constraint_types": [_ifeval_type(x) for x in sample.get("instruction_id_list", [])],
        "result_status": "fallback_verifier",
    }


def score_json_schema(sample: dict[str, Any], output: str) -> dict[str, Any]:
    parsed = extract_json_object(output)
    parse_valid = parsed is not None
    schema = sample["reference"]["schema"]
    expected = sample["reference"].get("expected") or {}
    schema_errors: list[str] = []
    schema_valid = False
    field_accuracy = 0.0
    if parse_valid:
        validator = Draft7Validator(schema)
        schema_errors = [error.message for error in sorted(validator.iter_errors(parsed), key=str)]
        schema_valid = not schema_errors
        field_accuracy = _field_accuracy(parsed, expected)
    return {
        "prediction": parsed,
        "gold": expected,
        "is_correct": bool(parse_valid and schema_valid),
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "field_accuracy": field_accuracy,
        "schema_errors": schema_errors,
        "constraint_types": sample["reference"].get("constraint_types", []),
        "result_status": "ok",
    }


def score_spider(sample: dict[str, Any], output: str) -> dict[str, Any]:
    sql = extract_sql(output)
    gold_sql = sample.get("reference", {}).get("gold_sql") or ""
    db_path = sample.get("db_path") or sample.get("reference", {}).get("db_path")
    exact_match = normalize_sql(sql) == normalize_sql(gold_sql) if sql and gold_sql else False
    execution = execute_sql_pair(db_path, sql, gold_sql) if db_path else {
        "execution_available": False,
        "execution_correct": False,
        "error": "missing db_path",
    }
    return {
        "prediction": sql,
        "gold": gold_sql,
        "is_correct": bool(execution.get("execution_correct")),
        "parse_valid": bool(sql),
        "exact_match": exact_match,
        "execution_available": bool(execution.get("execution_available")),
        "execution_correct": bool(execution.get("execution_correct")),
        "execution_error": execution.get("error"),
        "result_status": "ok" if execution.get("execution_available") else "execution_unavailable",
    }


def summarize_track5_rows(
    rows: list[dict[str, Any]],
    dataset_name: str,
    model_name: str,
    runtime_s: float,
) -> dict[str, Any]:
    if dataset_name == "mt_bench":
        metric_name = "mt_bench_judge_score"
        scored = [row for row in rows if (row.get("score") or {}).get("judge_score") is not None]
        metric_value = (
            sum(float(row["score"]["judge_score"]) for row in scored) / len(scored) if scored else None
        )
        result_status = "pending_judge" if not scored else "ok"
        extras = {"num_judged": len(scored), "avg_output_words": _avg_output_words(rows)}
    elif dataset_name == "ifeval":
        metric_name = "ifeval_pass_rate"
        metric_value = accuracy([{"is_correct": (row.get("score") or {}).get("is_correct")} for row in rows])
        result_status = "fallback_verifier"
        extras = _ifeval_breakdown(rows)
    elif dataset_name == "json_schema":
        metric_name = "json_schema_validity"
        metric_value = _mean_score(rows, "schema_valid")
        result_status = "ok"
        extras = {
            "json_parse_validity": _mean_score(rows, "parse_valid"),
            "json_schema_validity": metric_value,
            "json_field_level_accuracy": _mean_score(rows, "field_accuracy"),
            "constraint_breakdown": _constraint_breakdown(rows, "schema_valid"),
        }
    elif dataset_name == "spider":
        metric_name = "sql_execution_accuracy"
        metric_value = _mean_score(rows, "execution_correct")
        unavailable = sum(1 for row in rows if not (row.get("score") or {}).get("execution_available"))
        result_status = "partial_no_execution" if unavailable else "ok"
        extras = {
            "sql_execution_accuracy": metric_value,
            "sql_exact_match": _mean_score(rows, "exact_match"),
            "execution_unavailable_rate": unavailable / len(rows) if rows else 0.0,
        }
    else:
        raise ValueError(f"Unsupported Track 5 dataset: {dataset_name}")

    parser_failures = sum(1 for row in rows if _parser_failed(row, dataset_name))
    metric = {
        "track": "track5_constraints",
        "dataset": dataset_name,
        "model": model_name,
        "num_samples": len(rows),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "parser_failure_rate": parser_failures / len(rows) if rows else 0.0,
        "runtime_s": runtime_s,
        "result_status": result_status,
        "num_errors": sum(1 for row in rows if row.get("error")),
        "quality_validity_tradeoff": quality_validity_tradeoff(rows, dataset_name),
    }
    metric.update(extras)
    return metric


def quality_validity_tradeoff(rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    if dataset_name == "mt_bench":
        return {"quality_metric": "judge_score", "quality": None, "validity": None, "status": "pending_judge"}
    if dataset_name == "json_schema":
        return {
            "quality_metric": "field_level_accuracy",
            "quality": _mean_score(rows, "field_accuracy"),
            "validity_metric": "schema_validity",
            "validity": _mean_score(rows, "schema_valid"),
        }
    if dataset_name == "ifeval":
        return {
            "quality_metric": "not_applicable",
            "quality": None,
            "validity_metric": "ifeval_pass_rate",
            "validity": accuracy([{"is_correct": (row.get("score") or {}).get("is_correct")} for row in rows]),
        }
    if dataset_name == "spider":
        return {
            "quality_metric": "sql_exact_match",
            "quality": _mean_score(rows, "exact_match"),
            "validity_metric": "sql_execution_accuracy",
            "validity": _mean_score(rows, "execution_correct"),
        }
    return {}


def extract_json_object(text: str) -> Any | None:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        block = _first_balanced_json_object(candidate)
        if block:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return None


def extract_sql(text: str) -> str:
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    source = fenced[0] if fenced else text
    source = source.strip()
    match = re.search(r"(?is)\b(select|with)\b.*", source)
    if match:
        source = match.group(0)
    source = source.strip().strip("`")
    if ";" in source:
        source = source[: source.index(";") + 1]
    return source.strip()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())


def execute_sql_pair(db_path: str | Path, pred_sql: str, gold_sql: str) -> dict[str, Any]:
    if not pred_sql or not gold_sql:
        return {"execution_available": bool(db_path), "execution_correct": False, "error": "empty sql"}
    path = Path(db_path)
    if not path.exists():
        return {"execution_available": False, "execution_correct": False, "error": f"missing db: {path}"}
    conn = sqlite3.connect(path)
    try:
        t0 = time.perf_counter()
        pred = conn.execute(pred_sql).fetchall()
        gold = conn.execute(gold_sql).fetchall()
        return {
            "execution_available": True,
            "execution_correct": sorted(pred) == sorted(gold),
            "runtime_s": time.perf_counter() - t0,
            "error": None,
        }
    except Exception as exc:
        return {"execution_available": True, "execution_correct": False, "error": repr(exc)}
    finally:
        conn.close()


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _field_accuracy(pred: Any, expected: Any) -> float:
    leaves = list(_flatten(expected))
    if not leaves:
        return 0.0
    correct = 0
    for path, gold in leaves:
        if _get_path(pred, path) == gold:
            correct += 1
    return correct / len(leaves)


def _flatten(value: Any, prefix: tuple[Any, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, prefix + (key,))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _flatten(child, prefix + (idx,))
    else:
        yield prefix, value


def _get_path(value: Any, path: tuple[Any, ...]) -> Any:
    cur = value
    for part in path:
        if isinstance(part, int) and isinstance(cur, list) and part < len(cur):
            cur = cur[part]
        elif isinstance(part, str) and isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _mean_score(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    values = []
    for row in rows:
        value = (row.get("score") or {}).get(key)
        values.append(float(value) if value is not None else 0.0)
    return sum(values) / len(values)


def _parser_failed(row: dict[str, Any], dataset_name: str) -> bool:
    score = row.get("score") or {}
    if dataset_name == "json_schema":
        return not bool(score.get("parse_valid"))
    if dataset_name == "spider":
        return not bool(score.get("parse_valid"))
    return False


def _avg_output_words(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(len(str(row.get("model_output", "")).split()) for row in rows) / len(rows)


def _ifeval_type(instruction_id: str) -> str:
    iid = instruction_id.lower()
    if ":" in iid:
        return iid.split(":", 1)[0]
    if "/" in iid:
        return iid.split("/", 1)[0]
    return iid.split("_", 1)[0] if "_" in iid else iid


def _ifeval_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = {}
    supported = 0
    unsupported = 0
    loose_pass = []
    for row in rows:
        score = row.get("score") or {}
        fb = score.get("ifeval_fallback") or {}
        supported += int(fb.get("supported_count", 0))
        unsupported += int(fb.get("unsupported_count", 0))
        loose_pass.append(bool(fb.get("loose_supported_pass")))
        for result in fb.get("instruction_results", []):
            passed = result.get("passed")
            if passed is None:
                continue
            groups.setdefault(_ifeval_type(result["instruction_id"]), []).append(bool(passed))
    return {
        "loose_supported_pass_rate": sum(loose_pass) / len(loose_pass) if loose_pass else 0.0,
        "ifeval_supported_instructions": supported,
        "ifeval_unsupported_instructions": unsupported,
        "constraint_breakdown": {
            group: sum(values) / len(values) for group, values in sorted(groups.items())
        },
        "warning": "IFEval score uses fallback checker; run official IFEval for paper numbers.",
    }


def _constraint_breakdown(rows: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    groups: dict[str, list[bool]] = {}
    for row in rows:
        score = row.get("score") or {}
        for group in score.get("constraint_types") or []:
            groups.setdefault(group, []).append(bool(score.get(score_key)))
    return {group: sum(values) / len(values) for group, values in sorted(groups.items())}

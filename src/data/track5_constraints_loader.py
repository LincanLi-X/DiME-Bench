from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.utils.config import resolve_path
from src.utils.io import ensure_dir


def load_hf_dataset_with_fallbacks(dataset_cfg: dict[str, Any], cache_dir: str) -> Dataset:
    repo_ids = [dataset_cfg["repo_id"]] + list(dataset_cfg.get("fallback_repo_ids") or [])
    last_error: Exception | None = None
    for repo_id in repo_ids:
        try:
            subset = dataset_cfg.get("subset")
            split = dataset_cfg.get("split", "train")
            if subset:
                return load_dataset(repo_id, subset, split=split, cache_dir=cache_dir)
            return load_dataset(repo_id, split=split, cache_dir=cache_dir)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load dataset from {repo_ids}: {last_error!r}")


def normalize_track5_rows(dataset_name: str, raw: Dataset | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if dataset_name == "mt_bench":
        return [_normalize_mt_bench(i, row) for i, row in enumerate(raw)]
    if dataset_name == "ifeval":
        return [_normalize_ifeval(i, row) for i, row in enumerate(raw)]
    if dataset_name == "spider":
        return [_normalize_spider(i, row) for i, row in enumerate(raw)]
    if dataset_name == "json_schema":
        return list(build_json_schema_tasks())
    raise ValueError(f"Unsupported Track 5 dataset: {dataset_name}")


def select_samples(
    rows: list[dict[str, Any]],
    sample_size: int | None,
    sample_seed: int,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection: dict[str, Any] = {
        "source_num_samples": len(rows),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "mode": "full",
    }
    if sample_size is not None and len(rows) > sample_size:
        rng = random.Random(sample_seed)
        selected_indices = sorted(rng.sample(range(len(rows)), sample_size))
        rows = [rows[idx] for idx in selected_indices]
        selection.update({"mode": "random_sample", "selected_indices": selected_indices})
    if limit is not None:
        rows = rows[:limit]
        selection["limit"] = limit
        selection["mode"] = selection["mode"] + "+limit"
    return rows, selection


def build_json_schema_tasks() -> list[dict[str, Any]]:
    tasks = [
        {
            "sample_id": "json_schema_0001",
            "instruction": (
                "Create a user profile for Ada Lovelace with username ada_l, age 36, "
                "active status true, and role admin."
            ),
            "schema": {
                "type": "object",
                "required": ["username", "age", "active", "role"],
                "additionalProperties": False,
                "properties": {
                    "username": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "active": {"type": "boolean"},
                    "role": {"type": "string", "enum": ["admin", "editor", "viewer"]},
                },
            },
            "expected": {"username": "ada_l", "age": 36, "active": True, "role": "admin"},
            "constraint_types": ["required_fields", "field_types", "enum"],
        },
        {
            "sample_id": "json_schema_0002",
            "instruction": (
                "Create an order summary with order_id A100, customer Lee, total 19.95, "
                "currency USD, and two item names: notebook and pen."
            ),
            "schema": {
                "type": "object",
                "required": ["order_id", "customer", "total", "currency", "items"],
                "additionalProperties": False,
                "properties": {
                    "order_id": {"type": "string"},
                    "customer": {"type": "string"},
                    "total": {"type": "number"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "JPY"]},
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string"},
                    },
                },
            },
            "expected": {
                "order_id": "A100",
                "customer": "Lee",
                "total": 19.95,
                "currency": "USD",
                "items": ["notebook", "pen"],
            },
            "constraint_types": ["required_fields", "field_types", "array", "enum"],
        },
        {
            "sample_id": "json_schema_0003",
            "instruction": (
                "Create a support ticket with id T-7, priority high, status open, "
                "and nested requester name Mira with email mira@example.com."
            ),
            "schema": {
                "type": "object",
                "required": ["ticket_id", "priority", "status", "requester"],
                "additionalProperties": False,
                "properties": {
                    "ticket_id": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "status": {"type": "string", "enum": ["open", "closed"]},
                    "requester": {
                        "type": "object",
                        "required": ["name", "email"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string", "format": "email"},
                        },
                    },
                },
            },
            "expected": {
                "ticket_id": "T-7",
                "priority": "high",
                "status": "open",
                "requester": {"name": "Mira", "email": "mira@example.com"},
            },
            "constraint_types": ["required_fields", "field_types", "nested_object", "enum"],
        },
        {
            "sample_id": "json_schema_0004",
            "instruction": (
                "Create a weather report for Boston with temperature_c 12.5, condition rainy, "
                "and alerts as an empty array."
            ),
            "schema": {
                "type": "object",
                "required": ["city", "temperature_c", "condition", "alerts"],
                "additionalProperties": False,
                "properties": {
                    "city": {"type": "string"},
                    "temperature_c": {"type": "number"},
                    "condition": {"type": "string", "enum": ["sunny", "rainy", "snowy", "cloudy"]},
                    "alerts": {"type": "array", "items": {"type": "string"}},
                },
            },
            "expected": {
                "city": "Boston",
                "temperature_c": 12.5,
                "condition": "rainy",
                "alerts": [],
            },
            "constraint_types": ["required_fields", "field_types", "array", "enum"],
        },
    ]
    return [_json_schema_row(task) for task in tasks]


def build_mt_bench_fixtures() -> list[dict[str, Any]]:
    rows = [
        {
            "sample_id": "mt_bench_fixture_0001",
            "dataset": "mt_bench",
            "task_type": "open_generation",
            "question": "Explain why a fixed random seed matters in benchmark sampling.",
            "turn": 1,
            "category": "writing",
            "reference": None,
            "metadata": {"fixture": True},
        },
        {
            "sample_id": "mt_bench_fixture_0002",
            "dataset": "mt_bench",
            "task_type": "open_generation",
            "question": "Give two concise arguments for using execution accuracy in SQL generation.",
            "turn": 1,
            "category": "reasoning",
            "reference": None,
            "metadata": {"fixture": True},
        },
    ]
    return rows


def build_spider_fixtures() -> list[dict[str, Any]]:
    db_dir = ensure_dir(resolve_path("data/processed/track5_constraints/spider/fixture_dbs"))
    db_path = db_dir / "company.sqlite"
    _create_company_db(db_path)
    schema = "employees(id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)"
    return [
        {
            "sample_id": "spider_fixture_0001",
            "dataset": "spider",
            "task_type": "sql_generation",
            "db_id": "company",
            "db_path": str(db_path),
            "schema": schema,
            "question": "List the names of employees in the engineering department.",
            "reference": {
                "gold_sql": "SELECT name FROM employees WHERE department = 'engineering'",
                "db_id": "company",
                "db_path": str(db_path),
            },
            "metadata": {"fixture": True},
        },
        {
            "sample_id": "spider_fixture_0002",
            "dataset": "spider",
            "task_type": "sql_generation",
            "db_id": "company",
            "db_path": str(db_path),
            "schema": schema,
            "question": "How many employees have salary greater than 100000?",
            "reference": {
                "gold_sql": "SELECT COUNT(*) FROM employees WHERE salary > 100000",
                "db_id": "company",
                "db_path": str(db_path),
            },
            "metadata": {"fixture": True},
        },
    ]


def _normalize_mt_bench(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    turns = row.get("turns") or row.get("prompt") or row.get("question") or row.get("instruction")
    if isinstance(turns, list):
        question = str(turns[0])
    else:
        question = str(turns)
    sample_id = row.get("question_id") or row.get("id") or idx
    return {
        "sample_id": str(sample_id),
        "dataset": "mt_bench",
        "task_type": "open_generation",
        "question": question,
        "turn": int(row.get("turn", 1) or 1),
        "category": row.get("category"),
        "reference": None,
        "metadata": {"raw": dict(row)},
    }


def _normalize_ifeval(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row.get("key", idx)),
        "dataset": "ifeval",
        "task_type": "instruction_constraints",
        "prompt": str(row["prompt"]),
        "instruction_id_list": list(row.get("instruction_id_list", [])),
        "kwargs": row.get("kwargs", []),
        "reference": {
            "instruction_id_list": list(row.get("instruction_id_list", [])),
            "kwargs": row.get("kwargs", []),
        },
        "metadata": {"raw": dict(row)},
    }


def _normalize_spider(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    db_id = str(row.get("db_id") or row.get("database_id") or "")
    schema = _schema_text(row)
    gold_sql = str(row.get("query") or row.get("sql") or row.get("gold_sql") or "")
    db_path = row.get("db_path") or row.get("database_path")
    return {
        "sample_id": str(row.get("question_id", idx)),
        "dataset": "spider",
        "task_type": "sql_generation",
        "db_id": db_id,
        "db_path": str(db_path) if db_path else None,
        "schema": schema,
        "question": str(row.get("question") or row.get("utterance") or ""),
        "reference": {"gold_sql": gold_sql, "db_id": db_id, "db_path": str(db_path) if db_path else None},
        "metadata": {"raw": dict(row)},
    }


def _json_schema_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": task["sample_id"],
        "dataset": "json_schema",
        "task_type": "json_schema",
        "instruction": task["instruction"],
        "schema": task["schema"],
        "reference": {
            "expected": task["expected"],
            "schema": task["schema"],
            "constraint_types": task["constraint_types"],
        },
        "metadata": {"constraint_types": task["constraint_types"]},
    }


def _schema_text(row: dict[str, Any]) -> str:
    if row.get("schema"):
        return str(row["schema"])
    if row.get("db_schema"):
        return str(row["db_schema"])
    table_names = row.get("table_names_original") or row.get("table_names")
    column_names = row.get("column_names_original") or row.get("column_names")
    if table_names and column_names:
        tables: dict[int, list[str]] = {idx: [] for idx, _ in enumerate(table_names)}
        for item in column_names:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                table_idx, column = item
                if int(table_idx) >= 0:
                    tables.setdefault(int(table_idx), []).append(str(column))
        parts = []
        for idx, table in enumerate(table_names):
            cols = ", ".join(tables.get(idx) or ["*"])
            parts.append(f"{table}({cols})")
        return "\n".join(parts)
    return ""


def _create_company_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS employees")
        conn.execute(
            "CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)"
        )
        conn.executemany(
            "INSERT INTO employees(name, department, salary) VALUES (?, ?, ?)",
            [
                ("Ada", "engineering", 125000),
                ("Grace", "engineering", 99000),
                ("Lin", "sales", 110000),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def dump_schema_json(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2)

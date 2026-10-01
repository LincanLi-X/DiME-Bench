#!/usr/bin/env python3
"""Independently validate sealed results and optionally reproduce paper outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dimebench.artifacts.hashing import write_json_atomic
from dimebench.summarizers import discover_run_dirs, evaluate_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    evaluations = tuple(evaluate_run(path) for path in discover_run_dirs(args.run_root))
    payload = {
        "schema_version": "1.0",
        "status": "passed"
        if all(item.status == "passed" for item in evaluations)
        else "failed",
        "run_count": len(evaluations),
        "evaluations": evaluations,
    }
    if args.output is not None:
        write_json_atomic(args.output, payload)
    print(
        json.dumps(
            payload,
            default=lambda value: value.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

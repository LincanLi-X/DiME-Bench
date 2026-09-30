#!/usr/bin/env python3
"""Validate prepared DiME-Bench records, hashes, and split locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dimebench.artifacts.hashing import write_json_atomic
from dimebench.datasets import validate_prepared_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "data" / "processed" / "dime_bench_v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_prepared_data(args.path)
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

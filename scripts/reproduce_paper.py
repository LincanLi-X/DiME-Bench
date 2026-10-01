#!/usr/bin/env python3
"""Generate manuscript result artifacts from a tracked experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dimebench.reporting import load_reproduction_config, reproduce_paper

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = load_reproduction_config(args.config, project_root=PROJECT_ROOT)
    report = reproduce_paper(
        config,
        run_root=args.run_root,
        output_dir=args.output_dir,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

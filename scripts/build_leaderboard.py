#!/usr/bin/env python3
"""Build the static leaderboard from a versioned result submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dimebench.publishing import build_leaderboard_site

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=(PROJECT_ROOT / "results/releases/v1.0.0/result-submission.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "leaderboard/static/v1.0.0",
    )
    args = parser.parse_args()
    report = build_leaderboard_site(args.submission, args.output_dir)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

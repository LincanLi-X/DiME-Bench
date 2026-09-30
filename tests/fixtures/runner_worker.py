#!/usr/bin/env python3
"""Deterministic subprocess fixture for runner isolation and retry tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from dimebench.artifacts.hashing import hash_file, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--score", required=True, type=float)
    parser.add_argument("--fail-first", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.json"
    if not checkpoint.exists():
        write_json_atomic(
            checkpoint,
            {"job_id": args.job_id, "progress": "prediction_saved"},
        )
    checkpoint_sha256 = hash_file(checkpoint)
    marker = args.output_dir / ".first-attempt-failed"
    if args.fail_first and not marker.exists():
        marker.touch()
        print("intentional transient failure after durable checkpoint")
        return 42
    if hash_file(checkpoint) != checkpoint_sha256:
        raise RuntimeError("retry changed the existing checkpoint")
    write_json_atomic(
        args.output_dir / "score.json",
        {
            "job_id": args.job_id,
            "score": args.score,
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    print(f"completed {args.job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

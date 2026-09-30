#!/usr/bin/env python3
"""Prepare a DiME-Bench dataset suite from its tracked registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from dimebench.datasets import prepare_suite

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="dime_bench_v1")
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "data" / "registry.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare_suite(
        args.registry,
        args.suite,
        args.output_dir,
        mode="full" if args.full else "sample",
        cache_root=args.cache_dir,
        seed=args.seed,
        shuffle=args.shuffle,
        sample_limit=args.sample_limit,
        force=args.force,
    )
    print(f"prepared {args.suite}: {result['suite_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

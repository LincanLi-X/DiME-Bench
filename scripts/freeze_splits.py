#!/usr/bin/env python3
"""Freeze an existing normalized JSONL split into deterministic order."""

from __future__ import annotations

import argparse
from pathlib import Path

from dimebench.datasets import NormalizedDataset, freeze_split, load_dataset_manifest
from dimebench.datasets.split_lock import write_split_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-limit", type=int)
    args = parser.parse_args()
    manifest = load_dataset_manifest(args.manifest)
    dataset = NormalizedDataset.read_jsonl(
        args.input,
        dataset_id=args.dataset_id,
        split=args.split,
    )
    frozen, lock = freeze_split(
        dataset,
        manifest.manifest_hash,
        seed=args.seed,
        shuffle=args.shuffle,
        sample_limit=args.sample_limit,
    )
    frozen.write_jsonl(args.output)
    write_split_lock(args.lock, lock)
    print(f"samples: {len(frozen)}")
    print(f"dataset_hash: {frozen.dataset_hash}")
    print(f"lock_hash: {lock.lock_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

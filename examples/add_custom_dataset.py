"""Create a valid normalized local dataset for a custom task."""

from __future__ import annotations

import argparse
from pathlib import Path

from dimebench.datasets import GenerationRecord, NormalizedDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example-data/test.jsonl"))
    args = parser.parse_args()
    sample = GenerationRecord(
        sample_id="example_qa.test.item-1",
        dataset_id="example_qa",
        split="test",
        source_id="item-1",
        instruction="Return the capital of France.",
        references=("Paris",),
    )
    dataset = NormalizedDataset("example_qa", "test", (sample,))
    dataset.write_jsonl(args.output)
    print(f"wrote {len(dataset)} sample; dataset_hash={dataset.dataset_hash}")


if __name__ == "__main__":
    main()

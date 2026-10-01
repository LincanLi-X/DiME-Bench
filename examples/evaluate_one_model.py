"""Run one CPU-only model configuration through the public CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from dimebench.cli.main import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/smoke.yaml"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("example-results"))
    parser.add_argument("--run-id", default="example-one-model")
    args = parser.parse_args()
    return main(
        [
            "run",
            "--config",
            str(args.config),
            "--set",
            f"run_id={args.run_id}",
            "--set",
            f"output_dir={args.output_dir}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(run())

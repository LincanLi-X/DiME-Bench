"""Minimal command-line entry point for DiME-Bench."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from dimebench.schemas.model import ModelFamily
from dimebench.version import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _default_data_registry() -> Path:
    source_tree = PROJECT_ROOT / "data" / "registry.yaml"
    if source_tree.is_file():
        return source_tree
    return Path(sys.prefix) / "share" / "dimebench" / "data" / "registry.yaml"


DEFAULT_DATA_REGISTRY = _default_data_registry()
DEFAULT_DATA_OUTPUT = Path("data") / "processed"


def _default_task_config_root() -> Path:
    source_tree = PROJECT_ROOT / "configs" / "tasks"
    if source_tree.is_dir():
        return source_tree
    return Path(sys.prefix) / "share" / "dimebench" / "configs" / "tasks"


DEFAULT_TASK_CONFIG_ROOT = _default_task_config_root()
DEFAULT_PROMPT_ROOT = PACKAGE_ROOT / "prompts"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="dimebench",
        description=(
            "Mechanism-aware evaluation for discrete diffusion language models."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command")
    config_parser = commands.add_parser(
        "config",
        help="inspect and validate benchmark configuration",
    )
    config_commands = config_parser.add_subparsers(dest="config_command")
    validate_parser = config_commands.add_parser(
        "validate",
        help="validate a YAML run configuration",
    )
    validate_parser.add_argument("path", type=Path, help="path to a YAML config")
    validate_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted config key; may be repeated",
    )
    prepare_parser = commands.add_parser(
        "prepare",
        help="download or prepare a deterministic dataset suite",
    )
    prepare_parser.add_argument("--suite", required=True, help="registered suite id")
    prepare_parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_DATA_REGISTRY,
        help="data registry YAML",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_OUTPUT,
        help="prepared data root",
    )
    prepare_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="raw download cache (full mode only)",
    )
    prepare_parser.add_argument(
        "--full",
        action="store_true",
        help="prepare full upstream data instead of bundled schema fixtures",
    )
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--shuffle", action="store_true")
    prepare_parser.add_argument("--sample-limit", type=int, default=None)
    prepare_parser.add_argument("--force", action="store_true")

    data_parser = commands.add_parser("data", help="inspect prepared benchmark data")
    data_commands = data_parser.add_subparsers(dest="data_command")
    data_validate_parser = data_commands.add_parser(
        "validate",
        help="validate normalized records, hashes, and split locks",
    )
    data_validate_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA_OUTPUT / "dime_bench_v1",
        help="prepared suite directory",
    )

    task_parser = commands.add_parser("task", help="inspect task prompt contracts")
    task_commands = task_parser.add_subparsers(dest="task_command")
    preview_parser = task_commands.add_parser(
        "preview",
        help="render final prompts without running a model",
    )
    preview_parser.add_argument("--task", required=True, help="registered task id")
    preview_parser.add_argument(
        "--samples",
        required=True,
        type=Path,
        help="normalized dataset JSONL",
    )
    preview_parser.add_argument(
        "--model-family",
        choices=("autoregressive", "diffusion", "both"),
        default="both",
    )
    preview_parser.add_argument("--sample-id", default=None)
    preview_parser.add_argument("--limit", type=int, default=None)
    preview_parser.add_argument("--output", type=Path, default=None)
    preview_parser.add_argument(
        "--task-config-root",
        type=Path,
        default=DEFAULT_TASK_CONFIG_ROOT,
    )
    preview_parser.add_argument(
        "--prompt-root",
        type=Path,
        default=DEFAULT_PROMPT_ROOT,
    )

    infer_parser = commands.add_parser(
        "infer",
        help="run resumable raw-output inference without evaluation",
    )
    infer_parser.add_argument("--config", required=True, type=Path)
    infer_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted config key; may be repeated",
    )
    infer_parser.add_argument("--cache-dir", type=Path, default=None)
    infer_parser.add_argument("--max-attempts", type=int, default=3)
    infer_parser.add_argument("--retry-backoff", type=float, default=0.0)
    infer_parser.add_argument(
        "--prompt-root",
        type=Path,
        default=DEFAULT_PROMPT_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DiME-Bench command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config" and args.config_command == "validate":
        from dimebench.config import ConfigError, load_config

        try:
            spec = load_config(args.path, args.overrides)
        except ConfigError as exc:
            parser.error(str(exc))
        print(f"valid: {args.path}")
        print(f"config_hash: {spec.config_hash}")
        return 0
    if args.command == "prepare":
        from dimebench.datasets import DatasetError, prepare_suite

        try:
            manifest = prepare_suite(
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
        except DatasetError as exc:
            parser.error(str(exc))
        print(f"prepared: {args.suite}")
        print(f"mode: {manifest['mode']}")
        print(f"datasets: {len(manifest['datasets'])}")
        print(f"suite_hash: {manifest['suite_hash']}")
        return 0
    if args.command == "data" and args.data_command == "validate":
        from dimebench.datasets import validate_prepared_data

        report = validate_prepared_data(args.path)
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.status == "passed" else 1
    if args.command == "task" and args.task_command == "preview":
        from dimebench.prompts import PromptError
        from dimebench.tasks import (
            TaskCatalog,
            TaskError,
            preview_task_file,
            write_prompt_previews,
        )

        families: tuple[ModelFamily, ...]
        if args.model_family == "both":
            families = ("autoregressive", "diffusion")
        else:
            families = (cast(ModelFamily, args.model_family),)
        try:
            catalog = TaskCatalog.from_roots(
                args.task_config_root,
                args.prompt_root,
            )
            prompts = preview_task_file(
                catalog.get(args.task),
                args.samples,
                families,
                sample_id=args.sample_id,
                limit=args.limit,
            )
        except (TaskError, PromptError) as exc:
            parser.error(str(exc))
        if args.output is not None:
            write_prompt_previews(args.output, prompts)
            print(f"wrote {len(prompts)} prompt previews: {args.output.resolve()}")
        else:
            for prompt in prompts:
                print(
                    json.dumps(
                        prompt.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        return 0
    if args.command == "infer":
        from dimebench.config import ConfigError, load_config
        from dimebench.datasets import DatasetError
        from dimebench.inference import (
            InferenceEngine,
            InferenceError,
            RetryPolicy,
            load_inference_dataset,
        )
        from dimebench.models import create_adapter
        from dimebench.prompts import PromptError
        from dimebench.tasks import TaskError

        try:
            spec = load_config(args.config, args.overrides)
            dataset = load_inference_dataset(spec)
            adapter = create_adapter(spec.model)
            result = InferenceEngine(
                spec,
                adapter,
                prompt_root=args.prompt_root,
                cache_root=args.cache_dir,
                retry_policy=RetryPolicy(
                    max_attempts=args.max_attempts,
                    backoff_seconds=args.retry_backoff,
                ),
            ).run(dataset)
        except (
            ConfigError,
            DatasetError,
            InferenceError,
            PromptError,
            TaskError,
            ValueError,
        ) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "run_dir": str(result.run_dir.resolve()),
                    "predictions": str(result.predictions_path.resolve()),
                    "trace": str(result.trace_path.resolve()),
                    "total_samples": result.total_samples,
                    "previously_completed": result.previously_completed,
                    "written_predictions": result.written_predictions,
                    "cache_hits": result.cache_hits,
                    "failed_predictions": result.failed_predictions,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

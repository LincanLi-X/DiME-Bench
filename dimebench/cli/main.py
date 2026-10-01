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


def _default_model_config_root() -> Path:
    source_tree = PROJECT_ROOT / "configs" / "models"
    if source_tree.is_dir():
        return source_tree
    return Path(sys.prefix) / "share" / "dimebench" / "configs" / "models"


DEFAULT_MODEL_CONFIG_ROOT = _default_model_config_root()


def _default_specification() -> Path:
    source_tree = PROJECT_ROOT / "specification" / "benchmark-v1.json"
    if source_tree.is_file():
        return source_tree
    return (
        Path(sys.prefix) / "share" / "dimebench" / "specification" / "benchmark-v1.json"
    )


DEFAULT_SPECIFICATION = _default_specification()


def _default_reproduction_config() -> Path:
    source_tree = (
        PROJECT_ROOT / "configs" / "experiments" / "aaai2027" / "paper_full.yaml"
    )
    if source_tree.is_file():
        return source_tree
    return (
        Path(sys.prefix)
        / "share"
        / "dimebench"
        / "configs"
        / "experiments"
        / "aaai2027"
        / "paper_full.yaml"
    )


DEFAULT_REPRODUCTION_CONFIG = _default_reproduction_config()


def _default_smoke_config() -> Path:
    source_tree = PROJECT_ROOT / "configs" / "smoke.yaml"
    if source_tree.is_file():
        return source_tree
    return Path(sys.prefix) / "share" / "dimebench" / "configs" / "smoke.yaml"


DEFAULT_SMOKE_CONFIG = _default_smoke_config()


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
    list_parser = commands.add_parser("list", help="list registered benchmark assets")
    list_commands = list_parser.add_subparsers(dest="list_command")
    list_models_parser = list_commands.add_parser(
        "models", help="list tracked models and adapters"
    )
    list_models_parser.add_argument(
        "--model-config-root",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_ROOT,
    )
    list_models_parser.add_argument(
        "--format", choices=("table", "json"), default="table"
    )
    list_tasks_parser = list_commands.add_parser(
        "tasks", help="list task, track, dataset, and primary metric"
    )
    list_tasks_parser.add_argument(
        "--task-config-root",
        type=Path,
        default=DEFAULT_TASK_CONFIG_ROOT,
    )
    list_tasks_parser.add_argument(
        "--prompt-root",
        type=Path,
        default=DEFAULT_PROMPT_ROOT,
    )
    list_tasks_parser.add_argument(
        "--format", choices=("table", "json"), default="table"
    )
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

    inspect_parser = commands.add_parser(
        "inspect", help="inspect benchmark inputs and prompt contracts"
    )
    inspect_commands = inspect_parser.add_subparsers(dest="inspect_command")
    inspect_prompt_parser = inspect_commands.add_parser(
        "prompt", help="render one prepared sample without loading a model"
    )
    inspect_prompt_parser.add_argument("--task", required=True)
    inspect_prompt_parser.add_argument("--sample-id", required=True)
    inspect_prompt_parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="normalized JSONL; otherwise discovered under --suite-dir",
    )
    inspect_prompt_parser.add_argument(
        "--suite-dir",
        type=Path,
        default=DEFAULT_DATA_OUTPUT / "dime_bench_v1",
    )
    inspect_prompt_parser.add_argument(
        "--model-family",
        choices=("autoregressive", "diffusion", "both"),
        default="both",
    )
    inspect_prompt_parser.add_argument("--output", type=Path, default=None)
    inspect_prompt_parser.add_argument(
        "--task-config-root",
        type=Path,
        default=DEFAULT_TASK_CONFIG_ROOT,
    )
    inspect_prompt_parser.add_argument(
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
    run_parser = commands.add_parser(
        "run",
        help="run inference, evaluation, sealing, and summary export",
    )
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    run_parser.add_argument("--cache-dir", type=Path, default=None)
    run_parser.add_argument("--max-attempts", type=int, default=3)
    run_parser.add_argument("--retry-backoff", type=float, default=0.0)
    run_parser.add_argument(
        "--prompt-root",
        type=Path,
        default=DEFAULT_PROMPT_ROOT,
    )
    run_parser.add_argument("--report-dir", type=Path, default=None)
    run_parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    smoke_parser = commands.add_parser(
        "smoke",
        help="run the installed CPU-only Mock evaluation",
    )
    smoke_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    smoke_parser.add_argument("--run-id", default="dimebench-v1-smoke")
    smoke_parser.add_argument("--report-dir", type=Path, default=None)
    evaluate_parser = commands.add_parser(
        "evaluate",
        help="independently recompute and validate one sealed run",
    )
    evaluate_parser.add_argument("--run-dir", required=True, type=Path)
    evaluate_parser.add_argument("--output", type=Path, default=None)

    summarize_parser = commands.add_parser(
        "summarize",
        help="aggregate sealed runs and export report artifacts",
    )
    summarize_parser.add_argument("--run-dir", required=True, type=Path)
    summarize_parser.add_argument("--output-dir", type=Path, default=None)
    summarize_parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION,
    )

    reproduce_parser = commands.add_parser(
        "reproduce",
        help="reproduce tracked benchmark deliverables",
    )
    reproduce_commands = reproduce_parser.add_subparsers(dest="reproduce_command")
    paper_parser = reproduce_commands.add_parser(
        "paper",
        help="regenerate manuscript tables and plotting inputs",
    )
    paper_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REPRODUCTION_CONFIG,
    )
    paper_parser.add_argument("--run-root", type=Path, default=None)
    paper_parser.add_argument("--output-dir", type=Path, default=None)
    paper_parser.add_argument("--allow-incomplete", action="store_true")
    leaderboard_parser = commands.add_parser(
        "leaderboard",
        help="rank models from sealed runs or a benchmark summary",
    )
    leaderboard_source = leaderboard_parser.add_mutually_exclusive_group()
    leaderboard_source.add_argument("--run-dir", type=Path, default=None)
    leaderboard_source.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="versioned result-submission JSON (bare summaries are rejected)",
    )
    leaderboard_parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown"
    )
    leaderboard_parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DiME-Bench command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list" and args.list_command == "models":
        from dimebench.config import ConfigError, load_model_config

        try:
            models = [
                {
                    **load_model_config(path).model_dump(mode="json"),
                    "config": str(path),
                }
                for path in sorted(args.model_config_root.glob("*/*.yaml"))
            ]
        except ConfigError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(models, indent=2, sort_keys=True))
        else:
            print("MODEL_ID\tFAMILY\tADAPTER\tREVISION\tCONFIG")
            for model in models:
                print(
                    f"{model['id']}\t{model['family']}\t{model['adapter']}\t"
                    f"{model['revision']}\t{model['config']}"
                )
        return 0
    if args.command == "list" and args.list_command == "tasks":
        from dimebench.prompts import PromptError
        from dimebench.tasks import TaskCatalog, TaskError

        try:
            catalog = TaskCatalog.from_roots(
                args.task_config_root,
                args.prompt_root,
            )
        except (TaskError, PromptError) as exc:
            parser.error(str(exc))
        tasks = [
            {
                "id": task_id,
                "track": catalog.get(task_id).spec.track,
                "dataset_id": catalog.get(task_id).spec.dataset_id,
                "type": catalog.get(task_id).spec.type,
                "primary_metric": catalog.get(task_id).spec.primary_metric,
            }
            for task_id in catalog.ids
        ]
        if args.format == "json":
            print(json.dumps(tasks, indent=2, sort_keys=True))
        else:
            print("TASK_ID\tTRACK\tDATASET\tTYPE\tPRIMARY_METRIC")
            for task_row in tasks:
                print(
                    f"{task_row['id']}\t{task_row['track']}\t"
                    f"{task_row['dataset_id']}\t{task_row['type']}\t"
                    f"{task_row['primary_metric']}"
                )
        return 0
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
            data_manifest = prepare_suite(
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
        print(f"mode: {data_manifest['mode']}")
        print(f"datasets: {len(data_manifest['datasets'])}")
        print(f"suite_hash: {data_manifest['suite_hash']}")
        return 0
    if args.command == "data" and args.data_command == "validate":
        from dimebench.datasets import validate_prepared_data

        data_report = validate_prepared_data(args.path)
        print(json.dumps(data_report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if data_report.status == "passed" else 1
    if args.command == "task" and args.task_command == "preview":
        from dimebench.prompts import PromptError
        from dimebench.tasks import (
            TaskCatalog,
            TaskError,
            preview_task_file,
            write_prompt_previews,
        )

        preview_families: tuple[ModelFamily, ...]
        if args.model_family == "both":
            preview_families = ("autoregressive", "diffusion")
        else:
            preview_families = (cast(ModelFamily, args.model_family),)
        try:
            preview_catalog = TaskCatalog.from_roots(
                args.task_config_root,
                args.prompt_root,
            )
            preview_prompts = preview_task_file(
                preview_catalog.get(args.task),
                args.samples,
                preview_families,
                sample_id=args.sample_id,
                limit=args.limit,
            )
        except (TaskError, PromptError) as exc:
            parser.error(str(exc))
        if args.output is not None:
            write_prompt_previews(args.output, preview_prompts)
            print(
                f"wrote {len(preview_prompts)} prompt previews: {args.output.resolve()}"
            )
        else:
            for preview_prompt in preview_prompts:
                print(
                    json.dumps(
                        preview_prompt.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        return 0
    if args.command == "inspect" and args.inspect_command == "prompt":
        from dimebench.prompts import PromptError
        from dimebench.tasks import (
            TaskCatalog,
            TaskError,
            preview_task_file,
            write_prompt_previews,
        )

        inspect_families: tuple[ModelFamily, ...]
        if args.model_family == "both":
            inspect_families = ("autoregressive", "diffusion")
        else:
            inspect_families = (cast(ModelFamily, args.model_family),)
        try:
            inspect_catalog = TaskCatalog.from_roots(
                args.task_config_root,
                args.prompt_root,
            )
            inspect_task = inspect_catalog.get(args.task)
            inspect_samples = args.samples
            if inspect_samples is None:
                inspect_candidates = sorted(
                    (args.suite_dir / inspect_task.spec.dataset_id).glob("*.jsonl")
                )
                inspect_samples = next(
                    (
                        path
                        for path in inspect_candidates
                        if any(
                            json.loads(line).get("sample_id") == args.sample_id
                            for line in path.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        )
                    ),
                    None,
                )
                if inspect_samples is None:
                    raise TaskError(
                        f"sample {args.sample_id!r} was not found under "
                        f"{args.suite_dir / inspect_task.spec.dataset_id}"
                    )
            inspect_prompts = preview_task_file(
                inspect_task,
                inspect_samples,
                inspect_families,
                sample_id=args.sample_id,
                limit=1,
            )
        except (OSError, ValueError, TaskError, PromptError) as exc:
            parser.error(str(exc))
        if args.output is not None:
            write_prompt_previews(args.output, inspect_prompts)
            print(
                f"wrote {len(inspect_prompts)} prompt previews: {args.output.resolve()}"
            )
        else:
            for inspect_prompt in inspect_prompts:
                print(
                    json.dumps(
                        inspect_prompt.model_dump(mode="json"),
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
            inference_spec = load_config(args.config, args.overrides)
            inference_dataset = load_inference_dataset(inference_spec)
            inference_adapter = create_adapter(inference_spec.model)
            inference_result = InferenceEngine(
                inference_spec,
                inference_adapter,
                prompt_root=args.prompt_root,
                cache_root=args.cache_dir,
                retry_policy=RetryPolicy(
                    max_attempts=args.max_attempts,
                    backoff_seconds=args.retry_backoff,
                ),
            ).run(inference_dataset)
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
                    "run_id": inference_result.run_id,
                    "run_dir": str(inference_result.run_dir.resolve()),
                    "predictions": str(inference_result.predictions_path.resolve()),
                    "trace": str(inference_result.trace_path.resolve()),
                    "total_samples": inference_result.total_samples,
                    "previously_completed": inference_result.previously_completed,
                    "written_predictions": inference_result.written_predictions,
                    "cache_hits": inference_result.cache_hits,
                    "failed_predictions": inference_result.failed_predictions,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "smoke":
        smoke_report_dir = args.report_dir or args.output_dir / args.run_id / "reports"
        return main(
            [
                "run",
                "--config",
                str(DEFAULT_SMOKE_CONFIG),
                "--set",
                f"run_id={args.run_id}",
                "--set",
                f"output_dir={args.output_dir}",
                "--report-dir",
                str(smoke_report_dir),
            ]
        )
    if args.command == "run":
        from dimebench.config import ConfigError, load_config
        from dimebench.datasets import DatasetError
        from dimebench.inference import InferenceError, RetryPolicy
        from dimebench.prompts import PromptError
        from dimebench.reporting import PaperReproductionConfig, reproduce_paper
        from dimebench.tasks import TaskError
        from dimebench.workflow import WorkflowError, run_configured_workflow

        try:
            workflow_spec = load_config(args.config, args.overrides)
            workflow_result = run_configured_workflow(
                workflow_spec,
                prompt_root=args.prompt_root,
                cache_root=args.cache_dir,
                retry_policy=RetryPolicy(
                    max_attempts=args.max_attempts,
                    backoff_seconds=args.retry_backoff,
                ),
            )
            report_dir = args.report_dir or Path(workflow_result.run_dir) / "reports"
            workflow_summary = reproduce_paper(
                PaperReproductionConfig(
                    experiment_id=f"{workflow_result.run_id}-summary",
                    run_root=Path(workflow_result.run_dir),
                    output_dir=report_dir,
                    specification=args.specification.resolve(),
                    require_complete_tables=False,
                ),
                allow_incomplete=True,
            )
        except (
            ConfigError,
            DatasetError,
            InferenceError,
            PromptError,
            TaskError,
            WorkflowError,
            OSError,
            ValueError,
        ) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run": workflow_result.model_dump(mode="json"),
                    "summary": workflow_summary.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "evaluate":
        from dimebench.artifacts import load_manifest
        from dimebench.artifacts.hashing import write_json_atomic
        from dimebench.schemas import RunSpec
        from dimebench.summarizers import RunEvaluation, evaluate_run
        from dimebench.workflow import (
            EvaluationResult,
            WorkflowError,
            evaluate_configured_run,
        )

        try:
            run_manifest = load_manifest(args.run_dir / "run_manifest.json")
            evaluation_report: EvaluationResult | RunEvaluation
            if run_manifest.status == "running":
                evaluation_report = evaluate_configured_run(
                    RunSpec.model_validate(run_manifest.config)
                )
            else:
                evaluation_report = evaluate_run(args.run_dir)
        except (OSError, ValueError, WorkflowError) as exc:
            parser.error(str(exc))
        if args.output is not None:
            write_json_atomic(args.output, evaluation_report)
        print(
            json.dumps(
                evaluation_report.model_dump(mode="json"), indent=2, sort_keys=True
            )
        )
        return 0 if evaluation_report.status == "passed" else 1
    if args.command == "summarize":
        from dimebench.reporting import PaperReproductionConfig, reproduce_paper

        output_dir = args.output_dir or args.run_dir.parent / "summary"
        try:
            summary_report = reproduce_paper(
                PaperReproductionConfig(
                    experiment_id="cli-summary",
                    run_root=args.run_dir.resolve(),
                    output_dir=output_dir.resolve(),
                    specification=args.specification.resolve(),
                    require_complete_tables=False,
                ),
                allow_incomplete=True,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(summary_report.model_dump(mode="json"), indent=2, sort_keys=True)
        )
        return 0 if summary_report.status == "passed" else 1
    if args.command == "reproduce" and args.reproduce_command == "paper":
        from dimebench.reporting import load_reproduction_config, reproduce_paper

        try:
            reproduction_config = load_reproduction_config(
                args.config, project_root=Path.cwd()
            )
            reproduction_report = reproduce_paper(
                reproduction_config,
                run_root=args.run_root,
                output_dir=args.output_dir,
                allow_incomplete=args.allow_incomplete,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                reproduction_report.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if reproduction_report.status == "passed" else 1
    if args.command == "leaderboard":
        from dimebench.reporting import (
            leaderboard_from_runs,
            load_leaderboard_summary,
            render_leaderboard_markdown,
        )
        from dimebench.reporting._io import write_text_atomic

        default_summary = (
            Path("results") / "releases" / "v1.0.0" / "result-submission.json"
        )
        try:
            if args.run_dir is not None:
                leaderboard = leaderboard_from_runs(args.run_dir)
            else:
                summary_path = args.summary or default_summary
                if not summary_path.is_file():
                    print(
                        "No leaderboard summary found. Pass --run-dir or --summary "
                        "after completing a benchmark run."
                    )
                    return 0
                leaderboard = load_leaderboard_summary(summary_path)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        payload = (
            json.dumps(leaderboard.model_dump(mode="json"), indent=2, sort_keys=True)
            if args.format == "json"
            else render_leaderboard_markdown(leaderboard)
        )
        if args.output is not None:
            write_text_atomic(args.output, payload)
            print(f"wrote leaderboard: {args.output.resolve()}")
        else:
            print(payload)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

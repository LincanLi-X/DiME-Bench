"""Minimal real-model gate spanning all four DiME-Bench tracks."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, JsonValue, model_validator

from dimebench.artifacts import (
    MetricStore,
    PredictionStore,
    RunPaths,
    SampleMetricRecord,
    finalize_run,
    load_manifest,
    validate_run_artifacts,
)
from dimebench.artifacts.hashing import canonical_json, hash_file, hash_json
from dimebench.config import load_model_config, load_yaml
from dimebench.config.validation import validate_run_spec
from dimebench.datasets import NormalizedDataset, SampleRecord
from dimebench.datasets.base import load_dataset_manifest
from dimebench.evaluators import StandardEvaluator, evaluate_sample
from dimebench.evaluators.mechanism import (
    BoundaryConsistencyEvaluator,
    ContradictionRateEvaluator,
    ContradictionReductionEvaluator,
    EditSuccessEvaluator,
    OverEditRateEvaluator,
    PreservationScoreEvaluator,
    TransformersNLIBackend,
)
from dimebench.evaluators.standard import create_standard_evaluator
from dimebench.postprocessors import (
    postprocess_predictions,
    write_postprocessed_results,
)
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.dataset import DatasetSpec
from dimebench.schemas.model import ModelFamily, ModelSpec
from dimebench.schemas.run import DecodingSpec, RunSpec
from dimebench.schemas.task import TaskSpec


class TrackValidationCase(StrictSchema):
    """One fixed dataset selected for the minimal gate of a benchmark track."""

    track: Literal[
        "track1_general",
        "track2_infilling",
        "track3_editing",
        "track4_reasoning",
    ]
    dataset_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    task_config: Path
    manifest: Path
    max_output_tokens: int = Field(gt=0)
    dllm_block_length: int = Field(gt=0)
    mask_policy: Literal["append", "bounded_middle"]


class Step12Matrix(StrictSchema):
    """Auditable 4-track × 2-family Step 12 acceptance matrix."""

    schema_version: Literal["1.0"] = "1.0"
    suite: Literal["step12_minimal"] = "step12_minimal"
    sample_count: Literal[25] = 25
    dllm_nfes: Literal[16] = 16
    seed: int = Field(default=0, ge=0)
    ar_model_config: Path
    dllm_model_config: Path
    nli_model_id: str = Field(min_length=1)
    nli_revision: str = Field(min_length=1)
    tracks: tuple[TrackValidationCase, ...]

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> Step12Matrix:
        expected = {
            "track1_general",
            "track2_infilling",
            "track3_editing",
            "track4_reasoning",
        }
        observed = {case.track for case in self.tracks}
        if observed != expected or len(self.tracks) != 4:
            raise ValueError("Step 12 matrix must select exactly one case per track")
        for case in self.tracks:
            if case.max_output_tokens % case.dllm_block_length != 0:
                raise ValueError(
                    f"{case.track} output length must divide into dLLM blocks"
                )
            block_count = case.max_output_tokens // case.dllm_block_length
            if self.dllm_nfes % block_count != 0:
                raise ValueError(
                    f"{case.track} 16 NFEs must divide across all dLLM blocks"
                )
            expected_policy = (
                "bounded_middle" if case.track == "track2_infilling" else "append"
            )
            if case.mask_policy != expected_policy:
                raise ValueError(
                    f"{case.track} requires mask_policy={expected_policy!r}"
                )
        return self


def _resolve_config_path(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def load_step12_matrix(path: str | Path, project_root: str | Path) -> Step12Matrix:
    """Load the tracked matrix and validate every referenced source asset."""
    root = Path(project_root).resolve()
    matrix = Step12Matrix.model_validate(load_yaml(path))
    referenced = [matrix.ar_model_config, matrix.dllm_model_config]
    for case in matrix.tracks:
        referenced.extend((case.task_config, case.manifest))
    missing = [
        str(value)
        for value in referenced
        if not _resolve_config_path(root, value).is_file()
    ]
    if missing:
        raise ValueError(
            f"Step 12 matrix references missing files: {', '.join(missing)}"
        )
    return matrix


def _model_spec(
    path: Path,
    *,
    name_or_path: str | None,
    device: str,
    block_length: int | None = None,
) -> ModelSpec:
    base = load_model_config(path)
    adapter_kwargs = dict(base.adapter_kwargs)
    if block_length is not None:
        adapter_kwargs["block_length"] = block_length
    updates: dict[str, Any] = {
        "device": device,
        "adapter_kwargs": adapter_kwargs,
    }
    if name_or_path:
        updates["name_or_path"] = name_or_path
        updates["tokenizer_name_or_path"] = name_or_path
    return base.model_copy(update=updates)


def build_step12_run_specs(
    matrix: Step12Matrix,
    *,
    project_root: str | Path,
    prepared_root: str | Path,
    output_root: str | Path,
    ar_model: str | None = None,
    dllm_model: str | None = None,
    device: str = "cuda",
) -> tuple[RunSpec, ...]:
    """Resolve the fixed matrix into eight validated inference run specs."""
    root = Path(project_root).resolve()
    prepared = Path(prepared_root).resolve() / matrix.suite
    output = Path(output_root).resolve() / "runs"
    specs: list[RunSpec] = []
    for case in matrix.tracks:
        task_path = _resolve_config_path(root, case.task_config)
        task = TaskSpec.model_validate(load_yaml(task_path)).model_copy(
            update={"max_output_tokens": case.max_output_tokens}
        )
        manifest = load_dataset_manifest(_resolve_config_path(root, case.manifest))
        data_path = prepared / case.dataset_id / f"{case.split}.jsonl"
        if not data_path.is_file():
            raise ValueError(
                f"prepared Step 12 data is missing: {data_path}; run data preparation"
            )
        dataset = DatasetSpec(
            id=case.dataset_id,
            source="local",
            name_or_path=str(data_path),
            revision="step12_minimal",
            split=case.split,
            manifest_hash=manifest.manifest_hash,
            sample_limit=matrix.sample_count,
            shuffle=False,
            seed=matrix.seed,
        )
        for family, override in (
            ("autoregressive", ar_model),
            ("diffusion", dllm_model),
        ):
            is_diffusion = family == "diffusion"
            model_path = _resolve_config_path(
                root,
                matrix.dllm_model_config if is_diffusion else matrix.ar_model_config,
            )
            model = _model_spec(
                model_path,
                name_or_path=override,
                device=device,
                block_length=case.dllm_block_length if is_diffusion else None,
            )
            label = "dllm" if is_diffusion else "ar"
            decoding = DecodingSpec(
                max_new_tokens=case.max_output_tokens,
                temperature=0.0,
                do_sample=False,
                denoising_steps=matrix.dllm_nfes if is_diffusion else None,
                unmasking_strategy=("confidence_based" if is_diffusion else None),
                schedule="linear" if is_diffusion else None,
                mask_policy=case.mask_policy if is_diffusion else None,
            )
            specs.append(
                validate_run_spec(
                    RunSpec(
                        run_id=f"step12-{case.track}-{case.dataset_id}-{label}",
                        seed=matrix.seed,
                        deterministic=True,
                        batch_size=1,
                        output_dir=output,
                        model=model,
                        dataset=dataset,
                        task=task,
                        decoding=decoding,
                        tags=(
                            "step12",
                            "minimal_real_model_gate",
                            case.track,
                            "n25",
                        ),
                    )
                )
            )
    return tuple(specs)


class _EntailmentBoundaryBackend:
    """Entailment view over one shared NLI checkpoint instance."""

    backend_id = "transformers_nli_entailment"

    def __init__(self, backend: TransformersNLIBackend) -> None:
        self.backend = backend
        self.revision = backend.revision

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **self.backend.configuration(),
            "backend_id": self.backend_id,
            "positive_label": "entailment",
        }

    def score_pair(self, left: str, right: str) -> float:
        return self.backend.label_probability(left, right, "entailment")


def _evaluators_for(
    spec: RunSpec,
    *,
    nli_model_id: str,
    nli_revision: str,
    nli_device: str,
) -> tuple[StandardEvaluator, ...]:
    if spec.task.track == "track1_general":
        return (create_standard_evaluator("normalized_exact_match"),)
    if spec.task.track == "track2_infilling":
        nli = TransformersNLIBackend(
            nli_model_id,
            revision=nli_revision,
            device=nli_device,
        )
        return (
            create_standard_evaluator("token_f1"),
            BoundaryConsistencyEvaluator(_EntailmentBoundaryBackend(nli)),
            ContradictionRateEvaluator(nli),
        )
    if spec.task.track == "track3_editing":
        nli = TransformersNLIBackend(
            nli_model_id,
            revision=nli_revision,
            device=nli_device,
        )
        return (
            EditSuccessEvaluator(),
            PreservationScoreEvaluator(),
            OverEditRateEvaluator(),
            ContradictionReductionEvaluator(nli),
        )
    if spec.task.track == "track4_reasoning":
        return (create_standard_evaluator("option_accuracy"),)
    raise ValueError(f"unsupported Step 12 track: {spec.task.track}")


def _metric_fingerprint(records: Iterable[SampleMetricRecord]) -> str:
    payload = []
    for record in records:
        dumped = record.model_dump(mode="json", exclude_none=False)
        dumped.pop("created_at", None)
        payload.append(dumped)
    return hash_json(payload)


def _write_metric_records(path: Path, records: Iterable[SampleMetricRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            for record in records:
                stream.write(canonical_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _aggregate_signature(summary: Any) -> dict[str, Any]:
    return {
        metric_id: aggregate.model_dump(mode="json", exclude_none=False)
        for metric_id, aggregate in sorted(summary.metrics.items())
    }


def evaluate_step12_run(
    spec: RunSpec,
    *,
    nli_model_id: str,
    nli_revision: str,
    nli_device: str = "cuda",
) -> dict[str, Any]:
    """Score one run and prove scores can be regenerated from predictions."""
    paths = RunPaths(Path(spec.output_dir) / spec.run_id)
    manifest = load_manifest(paths.manifest)
    dataset = NormalizedDataset.read_jsonl(
        spec.dataset.name_or_path,
        dataset_id=spec.dataset.id,
        split=spec.dataset.split,
    )
    selected = tuple(dataset.records[: cast(int, spec.dataset.sample_limit)])
    if len(selected) != 25 or manifest.sample_count != 25:
        raise ValueError(f"{spec.run_id} must contain exactly 25 fixed samples")
    sample_map: dict[str, SampleRecord] = {
        sample.sample_id: sample for sample in selected
    }
    store = PredictionStore(paths.predictions, spec.run_id, spec.config_hash)
    ordered_predictions = tuple(store.get(sample.sample_id) for sample in selected)
    if len(store) != 25 or any(
        item.status != "success" for item in ordered_predictions
    ):
        raise ValueError(f"{spec.run_id} requires 25 successful raw predictions")
    if spec.model.family == "diffusion":
        if spec.decoding.denoising_steps != 16:
            raise ValueError(f"{spec.run_id} is not configured for exactly 16 NFEs")
        for prediction in ordered_predictions:
            request_metadata = prediction.request.get("metadata")
            inference_metadata = prediction.decoding_metadata.get("inference")
            usage = (
                inference_metadata.get("usage")
                if isinstance(inference_metadata, dict)
                else None
            )
            if (
                not isinstance(request_metadata, dict)
                or request_metadata.get("denoising_steps") != 16
                or not isinstance(usage, dict)
                or usage.get("denoising_steps") != 16
            ):
                raise ValueError(
                    f"{spec.run_id} prediction {prediction.sample_id} lacks "
                    "16-NFE provenance"
                )

    evaluators = _evaluators_for(
        spec,
        nli_model_id=nli_model_id,
        nli_revision=nli_revision,
        nli_device=nli_device,
    )
    official_results = postprocess_predictions(
        ordered_predictions,
        sample_map,
        spec.task,
    )
    postprocessed_path = paths.root / "postprocessed-results.jsonl"
    write_postprocessed_results(postprocessed_path, official_results)
    official_records = tuple(
        evaluate_sample(result, sample_map[result.sample_id], evaluators)
        for result in official_results
    )

    official_store = MetricStore(
        paths.sample_metrics,
        spec.run_id,
        spec.config_hash,
    )
    if len(official_store) == 0:
        for record in official_records:
            official_store.append(record)
    else:
        stored = tuple(official_store.get(item.sample_id) for item in official_records)
        if _metric_fingerprint(stored) != _metric_fingerprint(official_records):
            raise ValueError(f"{spec.run_id} existing scores differ from recomputation")
    official_summary = official_store.write_summary(paths.summary)

    # Re-enter the full prediction -> parse -> metric path and persist its own
    # outputs.  Comparison excludes timestamps but includes prediction hashes,
    # evaluator versions/configs, scores, denominators, and contributing IDs.
    recomputed_dir = paths.root / "recomputed"
    recomputed_results = postprocess_predictions(
        ordered_predictions,
        sample_map,
        spec.task,
    )
    recomputed_results_path = recomputed_dir / "postprocessed-results.jsonl"
    write_postprocessed_results(recomputed_results_path, recomputed_results)
    recomputed_records = tuple(
        evaluate_sample(result, sample_map[result.sample_id], evaluators)
        for result in recomputed_results
    )
    recomputed_metrics_path = recomputed_dir / "sample_metrics.jsonl"
    _write_metric_records(recomputed_metrics_path, recomputed_records)
    recomputed_store = MetricStore(
        recomputed_metrics_path,
        spec.run_id,
        spec.config_hash,
    )
    recomputed_summary = recomputed_store.write_summary(recomputed_dir / "summary.json")
    score_fingerprint = _metric_fingerprint(official_records)
    recomputed_fingerprint = _metric_fingerprint(recomputed_records)
    recomputation_matches = (
        score_fingerprint == recomputed_fingerprint
        and hash_file(postprocessed_path) == hash_file(recomputed_results_path)
        and _aggregate_signature(official_summary)
        == _aggregate_signature(recomputed_summary)
    )
    if not recomputation_matches:
        raise ValueError(f"{spec.run_id} independent score recomputation differs")

    finalized = finalize_run(paths, "completed")
    validate_run_artifacts(paths)
    primary = official_summary.metrics[spec.task.primary_metric]
    track_aggregates: dict[str, float | None] = {}
    if spec.task.track == "track4_reasoning":
        track_aggregates["planning_average"] = primary.value
    report = {
        "run_id": spec.run_id,
        "track": spec.task.track,
        "dataset_id": spec.dataset.id,
        "model_id": spec.model.id,
        "model_family": spec.model.family,
        "sample_count": len(selected),
        "successful_prediction_count": len(ordered_predictions),
        "dllm_nfes": spec.decoding.denoising_steps,
        "primary_metric": spec.task.primary_metric,
        "primary_metric_value": primary.value,
        "metrics": _aggregate_signature(official_summary),
        "track_aggregates": track_aggregates,
        "prediction_sha256": hash_file(paths.predictions),
        "score_fingerprint": score_fingerprint,
        "recomputed_score_fingerprint": recomputed_fingerprint,
        "recomputation_matches": recomputation_matches,
        "manifest_status": finalized.status,
        "run_dir": str(paths.root),
    }
    report_path = paths.root / "evaluation-report.json"
    from dimebench.artifacts.hashing import write_json_atomic

    write_json_atomic(report_path, report)
    return report


def model_family_counts(reports: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return family counts for the top-level acceptance report."""
    counts: dict[str, int] = {"autoregressive": 0, "diffusion": 0}
    for report in reports:
        family = cast(ModelFamily, report["model_family"])
        counts[family] += 1
    return counts


__all__ = [
    "Step12Matrix",
    "TrackValidationCase",
    "build_step12_run_specs",
    "evaluate_step12_run",
    "load_step12_matrix",
    "model_family_counts",
]

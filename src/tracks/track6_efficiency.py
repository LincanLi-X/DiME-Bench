from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.track6_efficiency_loader import select_track6_samples
from src.metrics.track6_efficiency import (
    NA,
    compute_trajectory_metrics,
    score_track6_output,
    summarize_output_rows,
)
from src.models.hf_local import HFLocalGenerator
from src.utils.config import get_dataset_config, load_model_config, load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text, require_cuda
from src.utils.io import append_jsonl, ensure_dir, read_jsonl, write_json
from src.utils.seed import set_seed


TRACK = "track6_efficiency"


def load_track6_model_configs(track_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    configs = []
    for item in track_cfg["models"]:
        if isinstance(item, str):
            configs.append(load_model_config(item))
        elif isinstance(item, dict):
            cfg = dict(item)
            cfg["_config_path"] = "<inline:track6_efficiency>"
            configs.append(cfg)
        else:
            raise TypeError(f"Unsupported model config entry: {item!r}")
    return configs


def get_track6_model_config(track_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_name_norm = model_name.lower()
    for cfg in load_track6_model_configs(track_cfg):
        aliases = {
            str(cfg["name"]).lower(),
            Path(str(cfg.get("_config_path", ""))).stem.lower(),
            str(cfg["repo_id"]).split("/")[-1].lower(),
        }
        if model_name_norm in aliases:
            return cfg
    available = ", ".join(cfg["name"] for cfg in load_track6_model_configs(track_cfg))
    raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")


def load_processed_samples(dataset_name: str) -> list[dict[str, Any]]:
    path = resolve_path(Path("data/processed/track6_efficiency") / dataset_name / "samples.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Run scripts/prepare_track6_efficiency_datasets.py first."
        )
    return read_jsonl(path)


def build_prompt(sample: dict[str, Any], dataset_cfg: dict[str, Any], setting: dict[str, Any]) -> str:
    template_path = resolve_path(dataset_cfg["prompt_template"])
    template = template_path.read_text(encoding="utf-8")
    fields = dict(sample.get("prompt_fields") or {})
    fields["target_output_length"] = int(setting.get("output_length") or sample.get("target_output_length") or 512)
    return template.format(**fields)


def sweep_settings(track_cfg: dict[str, Any], sweep: str) -> list[dict[str, Any]]:
    if sweep == "all":
        settings = []
        for name in ("t", "length", "batch"):
            settings.extend(sweep_settings(track_cfg, name))
        return settings
    cfg = track_cfg["sweeps"][sweep]
    settings = []
    for value in cfg["values"]:
        if sweep == "t":
            settings.append(
                {
                    "sweep": "t",
                    "setting_id": f"T{value}_length{cfg['output_length']}_batch{cfg['batch_size']}",
                    "steps": int(value),
                    "output_length": int(cfg["output_length"]),
                    "batch_size": int(cfg["batch_size"]),
                }
            )
        elif sweep == "length":
            settings.append(
                {
                    "sweep": "length",
                    "setting_id": f"T{cfg['steps']}_length{value}_batch{cfg['batch_size']}",
                    "steps": int(cfg["steps"]),
                    "output_length": int(value),
                    "batch_size": int(cfg["batch_size"]),
                }
            )
        elif sweep == "batch":
            settings.append(
                {
                    "sweep": "batch",
                    "setting_id": f"T{cfg['steps']}_length{cfg['output_length']}_batch{value}",
                    "steps": int(cfg["steps"]),
                    "output_length": int(cfg["output_length"]),
                    "batch_size": int(value),
                }
            )
        else:
            raise ValueError(f"Unknown sweep: {sweep}")
    return settings


def run_track6_model_dataset(
    model_name: str,
    dataset_name: str,
    sweep: str = "t",
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    require_gpu: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    mock_model: bool = False,
) -> dict[str, Any]:
    if require_gpu and not (dry_run or mock_model):
        require_cuda()
    set_seed(42)
    track_cfg = load_track_config(TRACK)
    model_cfg = get_track6_model_config(track_cfg, model_name)
    dataset_cfg = get_dataset_config(track_cfg, dataset_name)
    samples = load_processed_samples(dataset_name)
    samples, sample_selection = select_track6_samples(
        samples,
        dataset_cfg,
        sample_size=sample_size,
        sample_seed=sample_seed,
        limit=limit,
    )

    safe_model = _safe_name(model_cfg["name"])
    output_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["output_dir"]) / safe_model))
    metric_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["metric_dir"]) / safe_model))
    log_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["log_dir"]) / safe_model))
    trajectory_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["trajectory_dir"]) / safe_model))
    output_path = output_dir / f"{dataset_name}.jsonl"
    metric_path = metric_dir / f"{dataset_name}.json"
    log_path = log_dir / f"{dataset_name}.jsonl"
    trajectory_path = trajectory_dir / f"{dataset_name}.jsonl"
    if overwrite:
        for path in (output_path, log_path, trajectory_path):
            if path.exists():
                path.unlink()
    elif output_path.exists():
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to rerun.")

    generation_cfg_base = dict(track_cfg["generation"])
    settings = sweep_settings(track_cfg, sweep)
    result_status = "dry_run" if dry_run else "mock" if mock_model else "ok"
    all_rows: list[dict[str, Any]] = []
    all_log_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    generator = None
    if not (dry_run or mock_model):
        generator = HFLocalGenerator(model_cfg, generation_cfg_base)
    try:
        for setting in settings:
            generation_cfg = dict(generation_cfg_base)
            generation_cfg["batch_size"] = setting["batch_size"]
            generation_cfg["steps"] = setting["steps"]
            max_new_tokens = int(setting["output_length"])
            prompts = [build_prompt(sample, dataset_cfg, setting) for sample in samples]
            if dry_run:
                rows, log_rows, trajectory_rows = _dry_run_rows(
                    samples, prompts, model_cfg, dataset_name, dataset_cfg, setting, result_status
                )
            elif mock_model:
                rows, log_rows, trajectory_rows = _mock_run_rows(
                    samples, prompts, model_cfg, dataset_name, dataset_cfg, setting, result_status
                )
            else:
                assert generator is not None
                rows, log_rows, trajectory_rows = _real_run_rows(
                    generator,
                    samples,
                    prompts,
                    model_cfg,
                    dataset_name,
                    dataset_cfg,
                    generation_cfg,
                    setting,
                    max_new_tokens,
                    result_status,
                )
            for row in rows:
                append_jsonl(output_path, row)
            for log_row in log_rows:
                append_jsonl(log_path, log_row)
            for trajectory_row in trajectory_rows:
                append_jsonl(trajectory_path, trajectory_row)
            all_rows.extend(rows)
            all_log_rows.extend(log_rows)
            all_trajectory_rows.extend(trajectory_rows)
    finally:
        if generator is not None:
            generator.close()

    summary = summarize_output_rows(all_rows, dataset_name)
    trajectory_summary = compute_trajectory_metrics(all_trajectory_rows, str(model_cfg.get("family")))
    runtime_s = time.perf_counter() - t0
    metric = {
        "track": TRACK,
        "dataset": dataset_name,
        "model": model_cfg["name"],
        "num_samples": len(all_rows),
        "sample_selection": sample_selection,
        "sweep": sweep,
        "settings": settings,
        "runtime_s": runtime_s,
        "result_status": result_status,
        **summary,
        **trajectory_summary,
        "system": _summarize_logs(all_log_rows),
        "cuda": cuda_summary(),
        "nvidia_smi": nvidia_smi_text() if not dry_run else "dry_run",
    }
    write_json(metric_path, metric)
    _update_aggregate_tables(track_cfg, all_log_rows, all_rows, all_trajectory_rows, model_cfg, dataset_name)
    return metric


def _real_run_rows(
    generator: HFLocalGenerator,
    samples: list[dict[str, Any]],
    prompts: list[str],
    model_cfg: dict[str, Any],
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    setting: dict[str, Any],
    max_new_tokens: int,
    result_status: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    log_rows = []
    trajectory_rows = []
    batch_size = int(setting["batch_size"])
    iterable = list(range(0, len(samples), batch_size))
    for start in tqdm(iterable, desc=f"{model_cfg['name']}:{dataset_name}:{setting['setting_id']}"):
        batch_samples = samples[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        _reset_peak_memory()
        _sync_cuda()
        t0 = time.perf_counter()
        try:
            outputs, batch_trajectories = _generate_batch_with_trajectory(
                generator,
                batch_samples,
                batch_prompts,
                model_cfg,
                generation_cfg,
                max_new_tokens,
                setting,
            )
            error = None
        except Exception as exc:
            outputs = [""] * len(batch_samples)
            batch_trajectories = []
            error = repr(exc)
        _sync_cuda()
        latency_s = time.perf_counter() - t0
        peak_memory_gb = _peak_memory_gb()
        generated_tokens = len(batch_samples) * max_new_tokens
        for sample, prompt, output in zip(batch_samples, batch_prompts, outputs):
            score = score_track6_output({**sample, "target_output_length": max_new_tokens}, dataset_name, output)
            row = _output_row(
                sample,
                prompt,
                output,
                score,
                model_cfg,
                dataset_name,
                dataset_cfg,
                setting,
                result_status,
                error,
            )
            rows.append(row)
        log_row = _log_row(
            model_cfg,
            dataset_name,
            setting,
            len(batch_samples),
            latency_s,
            generated_tokens,
            peak_memory_gb,
            result_status,
            error,
        )
        log_rows.append(log_row)
        trajectory_rows.extend(batch_trajectories)
    return rows, log_rows, trajectory_rows


def _generate_batch_with_trajectory(
    generator: HFLocalGenerator,
    samples: list[dict[str, Any]],
    prompts: list[str],
    model_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    max_new_tokens: int,
    setting: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if model_cfg.get("family") != "dllm" or getattr(generator.model.config, "mask_token_id", None) is None:
        return generator.generate(prompts, max_new_tokens=max_new_tokens), []
    return _generate_diffusion_with_trajectory(generator, samples, prompts, generation_cfg, max_new_tokens, setting)


def _generate_diffusion_with_trajectory(
    generator: HFLocalGenerator,
    samples: list[dict[str, Any]],
    prompts: list[str],
    generation_cfg: dict[str, Any],
    max_new_tokens: int,
    setting: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    rendered = [generator._render_prompt(prompt) for prompt in prompts]
    encoded = generator.tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(generation_cfg.get("max_input_tokens", 4096)),
    )
    encoded = {k: v.to(generator.model.device) for k, v in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    mask_token_id = int(getattr(generator.model.config, "mask_token_id"))
    eos_token_id = generator.tokenizer.eos_token_id
    pad_token_id = generator.tokenizer.pad_token_id or eos_token_id
    mask_block = torch.full(
        (encoded["input_ids"].shape[0], max_new_tokens),
        mask_token_id,
        dtype=encoded["input_ids"].dtype,
        device=encoded["input_ids"].device,
    )
    input_ids = torch.cat([encoded["input_ids"], mask_block], dim=1)
    attention_mask = torch.cat([encoded["attention_mask"], torch.ones_like(mask_block)], dim=1)
    mutable_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mutable_mask[:, prompt_width:] = True
    steps = int(setting["steps"])
    trajectory_rows: list[dict[str, Any]] = []
    model_type = str(getattr(generator.model.config, "model_type", "")).lower()

    with torch.no_grad():
        for step in range(max(1, steps)):
            current_mask = (input_ids == mask_token_id) & mutable_mask
            if not current_mask.any():
                break
            forward_attention_mask = None if model_type == "dream" else attention_mask
            model_out = generator.model(
                input_ids=input_ids,
                attention_mask=forward_attention_mask,
                use_cache=False,
            )
            logits = model_out.logits
            if model_type == "dream":
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            probs = F.softmax(logits.float(), dim=-1)
            confidence, candidates = probs.max(dim=-1)
            candidates = candidates.masked_fill(candidates == mask_token_id, eos_token_id or pad_token_id)
            remaining_by_row = current_mask.sum(dim=1)
            remaining_steps = max(1, steps - step)
            for row_idx in range(input_ids.shape[0]):
                positions = torch.nonzero(current_mask[row_idx], as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                num_transfer = max(1, int(torch.ceil(remaining_by_row[row_idx].float() / remaining_steps).item()))
                num_transfer = min(num_transfer, positions.numel())
                row_conf = confidence[row_idx, positions]
                selected = positions[torch.topk(row_conf, k=num_transfer).indices]
                input_ids[row_idx, selected] = candidates[row_idx, selected]
            suffix_ids = input_ids[:, prompt_width:].detach().cpu()
            suffix_mask = (suffix_ids == mask_token_id).detach().cpu()
            suffix_conf = confidence[:, prompt_width:].detach().float().cpu()
            for row_idx, sample in enumerate(samples):
                trajectory_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "setting_id": setting["setting_id"],
                        "step": step + 1,
                        "token_ids": suffix_ids[row_idx].tolist(),
                        "mask_status": suffix_mask[row_idx].tolist(),
                        "mask_count": int(suffix_mask[row_idx].sum().item()),
                        "finalized_count": int((~suffix_mask[row_idx]).sum().item()),
                        "mean_confidence": float(suffix_conf[row_idx].mean().item()),
                    }
                )

    outputs = []
    for row_idx in range(input_ids.shape[0]):
        suffix = input_ids[row_idx, prompt_width:].tolist()
        cleaned = []
        for tok in suffix:
            if tok in {eos_token_id, pad_token_id, mask_token_id}:
                break
            cleaned.append(tok)
        outputs.append(generator.tokenizer.decode(cleaned, skip_special_tokens=True).strip())
    return outputs, trajectory_rows


def _dry_run_rows(
    samples: list[dict[str, Any]],
    prompts: list[str],
    model_cfg: dict[str, Any],
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    setting: dict[str, Any],
    result_status: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for sample, prompt in zip(samples, prompts):
        score = {
            "metric_name": dataset_cfg["metric"],
            "metric_value": 0.0,
            "quality_score": 0.0,
            "parser_failed": False,
            "prediction": None,
            "gold": sample.get("reference"),
        }
        rows.append(_output_row(sample, prompt, "", score, model_cfg, dataset_name, dataset_cfg, setting, result_status))
    log_rows = [
        _log_row(model_cfg, dataset_name, setting, len(samples), 0.0, 0, 0.0, result_status, None)
    ]
    return rows, log_rows, []


def _mock_run_rows(
    samples: list[dict[str, Any]],
    prompts: list[str],
    model_cfg: dict[str, Any],
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    setting: dict[str, Any],
    result_status: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    rows = []
    trajectory_rows = []
    for sample, prompt in zip(samples, prompts):
        output = _mock_output(sample, dataset_name, int(setting["output_length"]))
        score = score_track6_output({**sample, "target_output_length": setting["output_length"]}, dataset_name, output)
        rows.append(_output_row(sample, prompt, output, score, model_cfg, dataset_name, dataset_cfg, setting, result_status))
        if model_cfg.get("family") == "dllm":
            trajectory_rows.extend(_mock_trajectory(sample, setting))
    latency_s = max(0.001, time.perf_counter() - t0)
    log_rows = [
        _log_row(
            model_cfg,
            dataset_name,
            setting,
            len(samples),
            latency_s,
            len(samples) * int(setting["output_length"]),
            0.0,
            result_status,
            None,
        )
    ]
    return rows, log_rows, trajectory_rows


def _output_row(
    sample: dict[str, Any],
    prompt: str,
    output: str,
    score: dict[str, Any],
    model_cfg: dict[str, Any],
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    setting: dict[str, Any],
    result_status: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "dataset": dataset_name,
        "model": model_cfg["name"],
        "prompt": prompt,
        "model_output": output,
        "parsed_output": score.get("prediction"),
        "reference": sample.get("reference"),
        "decoding_config": {
            "model_family": model_cfg.get("family"),
            "sweep": setting["sweep"],
            "setting_id": setting["setting_id"],
            "steps": setting["steps"] if model_cfg.get("family") == "dllm" else NA,
            "output_length": setting["output_length"],
            "batch_size": setting["batch_size"],
            "metric": dataset_cfg.get("metric"),
            "result_status": result_status,
        },
        "score": score,
        "error": error,
    }


def _log_row(
    model_cfg: dict[str, Any],
    dataset_name: str,
    setting: dict[str, Any],
    actual_batch_size: int,
    latency_s: float,
    generated_tokens: int,
    peak_memory_gb: float,
    result_status: str,
    error: str | None,
) -> dict[str, Any]:
    model_family = model_cfg.get("family")
    forward_passes = int(setting["steps"]) if model_family == "dllm" else int(setting["output_length"])
    return {
        "track": TRACK,
        "dataset": dataset_name,
        "model": model_cfg["name"],
        "model_type": model_family,
        "sweep": setting["sweep"],
        "setting_id": setting["setting_id"],
        "T": int(setting["steps"]) if model_family == "dllm" else NA,
        "output_length": int(setting["output_length"]),
        "batch_size": int(setting["batch_size"]),
        "actual_batch_size": actual_batch_size,
        "forward_passes": forward_passes,
        "latency_s": latency_s,
        "tokens_per_s": generated_tokens / latency_s if latency_s > 0 else 0.0,
        "examples_per_s": actual_batch_size / latency_s if latency_s > 0 else 0.0,
        "peak_memory_gb": peak_memory_gb,
        "generated_tokens": generated_tokens,
        "result_status": result_status,
        "error": error,
    }


def _summarize_logs(log_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not log_rows:
        return {}
    total_latency = sum(float(row.get("latency_s") or 0.0) for row in log_rows)
    total_tokens = sum(int(row.get("generated_tokens") or 0) for row in log_rows)
    total_examples = sum(int(row.get("actual_batch_size") or 0) for row in log_rows)
    return {
        "latency_s": total_latency,
        "tokens_per_s": total_tokens / total_latency if total_latency > 0 else 0.0,
        "examples_per_s": total_examples / total_latency if total_latency > 0 else 0.0,
        "peak_memory_gb": max(float(row.get("peak_memory_gb") or 0.0) for row in log_rows),
        "forward_passes": sum(int(row.get("forward_passes") or 0) for row in log_rows),
    }


def _update_aggregate_tables(
    track_cfg: dict[str, Any],
    log_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    dataset_name: str,
) -> None:
    out_dir = ensure_dir(resolve_path(track_cfg["reporting"]["metric_dir"]))
    quality_by_setting: dict[str, list[float]] = {}
    for row in output_rows:
        quality_by_setting.setdefault(row["decoding_config"]["setting_id"], []).append(
            float((row.get("score") or {}).get("quality_score", 0.0))
        )
    trajectory_by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        trajectory_by_setting.setdefault(str(row.get("setting_id")), []).append(row)

    table_rows = []
    for log in log_rows:
        setting_id = log["setting_id"]
        qualities = quality_by_setting.get(setting_id, [])
        traj_summary = compute_trajectory_metrics(trajectory_by_setting.get(setting_id, []), str(model_cfg.get("family")))
        table_rows.append(
            {
                **{k: log[k] for k in [
                    "model",
                    "model_type",
                    "dataset",
                    "sweep",
                    "T",
                    "output_length",
                    "batch_size",
                    "forward_passes",
                    "latency_s",
                    "tokens_per_s",
                    "examples_per_s",
                    "peak_memory_gb",
                    "result_status",
                ]},
                "quality_score": sum(qualities) / len(qualities) if qualities else 0.0,
                "str": traj_summary.get("str", NA),
                "foe": traj_summary.get("foe", NA),
                "mean_trc": traj_summary.get("mean_trc", NA),
            }
        )
    for sweep_name, filename in [
        ("t", "t_sweep_table.csv"),
        ("length", "length_sweep_table.csv"),
        ("batch", "batch_sweep_table.csv"),
    ]:
        rows = [row for row in table_rows if row["sweep"] == sweep_name]
        if rows:
            _upsert_csv(out_dir / filename, rows, key_fields=["model", "dataset", "sweep", "T", "output_length", "batch_size"])
    if table_rows:
        _upsert_csv(
            out_dir / "quality_compute_pareto.csv",
            table_rows,
            key_fields=["model", "dataset", "sweep", "T", "output_length", "batch_size"],
        )
        _upsert_csv(
            out_dir / "decoding_dynamics_table.csv",
            table_rows,
            key_fields=["model", "dataset", "sweep", "T", "output_length", "batch_size"],
        )


def _upsert_csv(path: Path, rows: list[dict[str, Any]], key_fields: list[str]) -> None:
    ensure_dir(path.parent)
    existing = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    incoming_keys = {_row_key(row, key_fields) for row in rows}
    kept = [row for row in existing if _row_key(row, key_fields) not in incoming_keys]
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)


def _row_key(row: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def _mock_output(sample: dict[str, Any], dataset_name: str, output_length: int) -> str:
    if dataset_name == "json_schema_generation":
        reference = sample.get("reference")
        if isinstance(reference, dict) and reference:
            import json

            return json.dumps(reference, ensure_ascii=False)
        return '{"name":"mock","role":"test","active":true}'
    if dataset_name == "fixed_length":
        topic = (sample.get("prompt_fields") or {}).get("topic", "benchmark")
        tokens = [str(topic).replace(" ", "_")] + ["token"] * max(0, min(output_length, 64) - 1)
        return " ".join(tokens)
    reference = sample.get("reference")
    return str(reference) if reference else "mock output"


def _mock_trajectory(sample: dict[str, Any], setting: dict[str, Any]) -> list[dict[str, Any]]:
    length = int(setting["output_length"])
    steps = int(setting["steps"])
    rows = []
    token_ids = [0] * length
    for step in range(1, steps + 1):
        finalized = min(length, max(1, int(length * step / steps)))
        for idx in range(finalized):
            token_ids[idx] = idx + 10
        mask_status = [idx >= finalized for idx in range(length)]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "setting_id": setting["setting_id"],
                "step": step,
                "token_ids": list(token_ids),
                "mask_status": mask_status,
                "mask_count": length - finalized,
                "finalized_count": finalized,
                "mean_confidence": min(0.99, 0.4 + 0.6 * step / steps),
            }
        )
    return rows


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_memory_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        pass
    return 0.0


def _safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name from Track 6 config, or 'all'.")
    parser.add_argument("--dataset", required=True, help="Dataset name from Track 6 config, or 'all'.")
    parser.add_argument("--sweep", choices=["t", "length", "batch", "all"], default="t")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config(TRACK)
    model_names = [cfg["name"] for cfg in load_track6_model_configs(track_cfg)] if args.model == "all" else [args.model]
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    metrics = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            metrics.append(
                run_track6_model_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    sweep=args.sweep,
                    limit=args.limit,
                    sample_size=args.sample_size,
                    sample_seed=args.sample_seed,
                    require_gpu=not args.no_require_cuda,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    mock_model=args.mock_model,
                )
            )
    print(metrics)


if __name__ == "__main__":
    main()

# Quick start

DiME-Bench supports Python 3.10-3.12. The smallest complete evaluation uses a
deterministic Mock dLLM, four bundled infilling examples, and no GPU.

## Install and verify

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

dimebench --version
dimebench smoke --output-dir outputs
```

The smoke command performs inference, postprocessing, metric evaluation,
artifact sealing, and report generation. A successful run creates
`outputs/dimebench-v1-smoke/` with four predictions and a completed manifest.

## Prepare benchmark data

```bash
dimebench prepare --suite dime_bench_v1
dimebench data validate data/processed/dime_bench_v1
dimebench list tasks
```

Sample mode is offline and intended for development. Use `--full` only after
reviewing dataset licenses and access requirements.

## Run one configured model

```bash
dimebench infer --config configs/smoke.yaml
dimebench evaluate --run-dir outputs/smoke_diffusion_infilling
dimebench summarize \
  --run-dir outputs/smoke_diffusion_infilling \
  --output-dir reports/smoke
```

For real models, install the appropriate extras and replace the Mock model
configuration. Raw predictions remain separate from scoring, so metrics can be
independently recomputed without rerunning the model.

## Next steps

- [Architecture and execution flow](../concepts/architecture.md)
- [Run and result formats](../concepts/results.md)
- [Four benchmark tracks](../tracks/overview.md)
- [Paper reproduction](../reproduction/paper.md)
- [Adding a model](../guides/add-model.md)

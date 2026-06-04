#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_track5_constraints_datasets.py --dataset json_schema --limit 2
python scripts/prepare_track5_constraints_datasets.py --dataset spider --limit 2 --allow-fixtures
python scripts/run_track5_constraints.py --model LLaDA-1.5 --dataset json_schema --limit 2 --mock-model --overwrite --no-require-cuda
python scripts/run_track5_constraints.py --model LLaDA-1.5 --dataset spider --limit 2 --mock-model --overwrite --no-require-cuda
python scripts/check_track5_constraints_setup.py --allow-missing-models --allow-missing-datasets

# Handover

**Last updated:** 2026-09-19

Week 9 project: **energy-forecasting-service** — ship the Week 5 TFT with MLOps (tracking first).

Remote: https://github.com/mehmetertac/energy-forecasting-service-.git

## What is done

| Area | Status |
|------|--------|
| Production layout (`src/`, `tests/`, `scripts/`, `docker/`, `dashboard/`, CI) | Done |
| Port Week 5 TFT + feature pipeline into `energy_forecasting` | Done — no MinT / N-HiTS |
| MLflow tracking (`params`, per-fold pinball/CRPS/coverage, artifacts) | Done — 3 local smoke runs (`tft-h16/32/64-smoke`) in `./mlruns` |
| W&B optional `--wandb` mirror | Wired; not executed (no `WANDB_API_KEY` on this machine) |
| Unit tests + pre-commit + GitHub Actions (no torch in CI) | Done |
| FastAPI serving | Stub only (`/health`, P10/P50/P90 schema) |
| Docker image | Placeholder |
| Streamlit dashboard | Placeholder |

## Goal

Take the Week 5 global TFT (`QuantileLoss` at 0.1 / 0.5 / 0.9, 168h encoder, 24h horizon) through a recruiter-facing service repo. This task: importable modules + experiment tracking. Later: Docker, FastAPI inference, Streamlit bands.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,train]"
python scripts/fetch_data.py
pytest tests/ -q
python scripts/train.py --n-splits 2 --max-epochs 2 --max-plants 2 --hidden-size 16 --limit-train-batches 20 --limit-val-batches 5
mlflow ui --backend-store-uri ./mlruns
```

Copy Week 5 raw cache instead of re-downloading:

```powershell
Copy-Item -Recurse ..\multi-site-solar-hierarchy\data\raw data\raw
python scripts/fetch_data.py
```

## Repo layout

| Path | Purpose |
|------|---------|
| `src/energy_forecasting/config.py` | Horizon, quantiles, TFT hparams, paths |
| `src/energy_forecasting/data/` | OPSD/Open-Meteo fetch, disaggregation, features |
| `src/energy_forecasting/model/tft.py` | `TFTForecaster` |
| `src/energy_forecasting/model/tracking.py` | MLflow + optional W&B |
| `src/energy_forecasting/api/schema.py` | `QuantileForecast` (pred_q10/50/90) |
| `scripts/train.py` | Rolling-origin CV CLI |
| `scripts/fetch_data.py` | Dataset assembly |
| `tests/` | Synthetic unit tests |
| `.github/workflows/tests.yml` | pytest on every push |
| `mlruns/` | Local MLflow store (gitignored) |
| `artifacts/` | Metrics, OOF parquet, checkpoints, plots |

## Core module API

```python
from energy_forecasting.data import build_dataset, load_plants_hourly, load_plants_metadata
from energy_forecasting.model.tft_features import prepare_tft_frame
from energy_forecasting.model.tft import TFTForecaster
from energy_forecasting.model.tracking import ExperimentTracker
from energy_forecasting.api.schema import QuantileForecast

# Quantiles stay [0.1, 0.5, 0.9]; predict_quantiles emits pred_q10/pred_q50/pred_q90
```

CLI flags on `scripts/train.py`: `--n-splits`, `--max-epochs`, `--hidden-size`, `--max-plants`, `--limit-train-batches`, `--limit-val-batches`, `--wandb`, `--run-name`.

## Tracking contract

Logged **params:** `horizon`, `hidden_size`, `quantiles`, `encoder_length`, `learning_rate`, `dropout`, `batch_size`, `max_epochs`, `n_splits`.

Logged **metrics (per fold + mean):** `pinball_q10`, `pinball_q50`, `pinball_q90`, `crps`, `pi_coverage`.

## Smoke tracking (2026-09-19)

Three 2-fold / 2-epoch / 2-plant runs, `hidden_size` ∈ {16, 32, 64}. Open with `mlflow ui --backend-store-uri ./mlruns`. Horizon-1 eval on smoke splits only merged 2 rows/fold — numbers are for the UI, not a production claim.

## Suggested next step

1. Set `WANDB_API_KEY` and re-run one smoke with `--wandb` to complete the hosted-vs-self-managed comparison live.
2. Next Week 9 task: Dockerized FastAPI inference endpoint returning P10/P50/P90.

## Key commit

`1218dc1` — scaffold the service repo, port the Week 5 TFT + feature pipeline, and add MLflow tracking.

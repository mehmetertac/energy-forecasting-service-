# Handover

**Last updated:** 2026-09-24

Week 9 project: **energy-forecasting-service** — ship the Week 5 TFT with MLOps (tracking first).

Remote: https://github.com/mehmetertac/energy-forecasting-service-.git

## What is done

| Area | Status |
|------|--------|
| Production layout (`src/`, `tests/`, `scripts/`, `docker/`, `dashboard/`, CI) | Done |
| Port Week 5 TFT + feature pipeline into `energy_forecasting` | Done — no MinT / N-HiTS |
| MLflow tracking (`params`, per-fold pinball/CRPS/coverage, artifacts) | Done — 3 local smoke runs (`tft-h16/32/64-smoke`) in `./mlruns` |
| MLflow Model Registry + production loader | Done — `tft-solar-quantile` Staging → Production via `scripts/register_model.py` |
| FastAPI batch inference | Done — `GET /health`, `GET /plants`, `GET /metrics`, `POST /forecast` (P10/P50/P90 + optional actual + model version) |
| W&B optional `--wandb` mirror | Wired; not executed (no `WANDB_API_KEY` on this machine) |
| Unit tests + pre-commit + GitHub Actions CI (no torch) | Done — calendar/solar feature values, model contract (finite ordered quantiles, horizon length), API schema + 422 validation; ruff + pytest + Docker build on every push/PR |
| Docker image + compose | Done — multi-stage serve image (~1.39 GB), `docker compose up` (API + MLflow + Streamlit dashboard) |
| Streamlit dashboard | Done — site selector, horizon slider, P50 + P10–P90 band over actuals, pinball/coverage panel |

## Goal

Take the Week 5 global TFT (`QuantileLoss` at 0.1 / 0.5 / 0.9, 168h encoder, 24h horizon) through a recruiter-facing service repo: importable modules, experiment tracking, registry-backed serving, Docker, and a Streamlit dashboard.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,train,serve]"
python scripts/fetch_data.py
pytest tests/ -q
python scripts/train.py --n-splits 2 --max-epochs 2 --max-plants 2 --hidden-size 16 --limit-train-batches 20 --limit-val-batches 5
python scripts/register_model.py
uvicorn energy_forecasting.api.app:app --host 127.0.0.1 --port 8000
mlflow ui --backend-store-uri ./mlruns
```

**Docker:**

```powershell
docker compose up --build
# seed Production model once — see docker/README.md
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
| `src/energy_forecasting/model/registry.py` | Model Registry, `get_production_model()`, honors `MLFLOW_TRACKING_URI` |
| `src/energy_forecasting/model/tracking.py` | MLflow + optional W&B |
| `src/energy_forecasting/api/app.py` | FastAPI `GET /health`, `GET /plants`, `GET /metrics`, `POST /forecast` |
| `src/energy_forecasting/api/schema.py` | Pydantic request/response (P10/P50/P90 + optional actual) |
| `dashboard/app.py` | Streamlit dashboard (reads API on :8501) |
| `dashboard/client.py` | HTTP client + latest-origin filter |
| `dashboard/Dockerfile` | Lean Streamlit image (no torch) |
| `scripts/train.py` | Rolling-origin CV CLI |
| `scripts/register_model.py` | Register best run → Production |
| `scripts/seed_docker_mlruns.py` | Seed `tft-solar-quantile` Production for Docker smoke |
| `scripts/fetch_data.py` | Dataset assembly |
| `Dockerfile` | Multi-stage serve image (uvicorn, non-root) |
| `docker-compose.yml` | API + MLflow server + Streamlit dashboard |
| `requirements.lock` | Pinned `[serve]` deps for Linux builds |
| `tests/` | Synthetic unit tests |
| `.github/workflows/ci.yml` | ruff, pytest, Docker build on every push/PR (required check) |
| `mlruns/` | Local MLflow store (gitignored) |
| `artifacts/` | Metrics, OOF parquet, checkpoints, plots |

## Core module API

```python
from energy_forecasting.data import build_dataset, load_plants_hourly, load_plants_metadata
from energy_forecasting.model.tft_features import prepare_tft_frame
from energy_forecasting.model.tft import TFTForecaster
from energy_forecasting.model.tracking import ExperimentTracker
from energy_forecasting.model.registry import get_production_model, register_best_model
from energy_forecasting.api.schema import ForecastRequest, ForecastResponse, QuantileForecast

# Production model (API uses this, not checkpoint paths):
info = get_production_model()  # uri: models:/tft-solar-quantile/Production
# Docker / compose: set MLFLOW_TRACKING_URI=http://mlflow:5000
```

CLI flags on `scripts/train.py`: `--n-splits`, `--max-epochs`, `--hidden-size`, `--max-plants`, `--limit-train-batches`, `--limit-val-batches`, `--wandb`, `--run-name`.

`scripts/register_model.py`: `--run-id`, `--metric` (default `mean_crps`), `--model-name`, `--no-promote`.

## Inference mode

**Batch (default).** Day-ahead forecasts are precomputed on a schedule and stored in the registered pyfunc artifact (OOF/day-ahead parquet). `POST /forecast` serves cached rows by `plant_id` + `horizon`. On-demand TFT at request time and streaming sub-hourly updates are documented in README as future paths.

## Tracking contract

Logged **params:** `horizon`, `hidden_size`, `quantiles`, `encoder_length`, `learning_rate`, `dropout`, `batch_size`, `max_epochs`, `n_splits`.

Logged **metrics (per fold + mean):** `pinball_q10`, `pinball_q50`, `pinball_q90`, `crps`, `pi_coverage`.

## Smoke tracking (2026-09-19)

Three 2-fold / 2-epoch / 2-plant runs, `hidden_size` ∈ {16, 32, 64}. Best by `mean_crps`: `tft-h16-smoke` → registered as `tft-solar-quantile` v1 Production.

## Suggested next step

1. Optional: set `WANDB_API_KEY` and re-run one smoke with `--wandb`.
2. Optional: on-demand TFT inference path (CPU torch in a separate image).

## Key commit

Streamlit dashboard + API `/plants`/`/metrics` + compose third service; Week 9 reflection close-out.

# energy-forecasting-service

[![ci](https://github.com/mehmetertac/energy-forecasting-service-/actions/workflows/ci.yml/badge.svg)](https://github.com/mehmetertac/energy-forecasting-service-/actions/workflows/ci.yml)

Recruiter-facing MLOps repo: take a **Week 5 Temporal Fusion Transformer** through experiment tracking, then (this week) Docker, FastAPI, CI, and a Streamlit demo.

The served contract is **probabilistic throughout**: **P10 / P50 / P90**. Point MAE is a side metric, not the product.

Ported from [`multi-site-solar-hierarchy`](https://github.com/mehmetertac/multi-site-solar-hierarchy) (TFT + feature pipeline). MinT reconciliation stays in that repo; this service ships the global plant-level TFT.

## Quick start

Requires **Python 3.11**.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,train]"
# optional notebook kernel: pip install -e ".[notebooks]"

# Reuse Week 5 raw cache if present, or download OPSD + Open-Meteo
python scripts/fetch_data.py

pytest tests/ -q

# Smoke train (logs to ./mlruns)
python scripts/train.py --n-splits 2 --max-epochs 2 --max-plants 2 --hidden-size 16 --limit-train-batches 20 --limit-val-batches 5 --run-name tft-h16-smoke

mlflow ui --backend-store-uri ./mlruns
```

Install git hooks once:

```powershell
pre-commit install
```

## What this means in dispatch / market terms

| Quantile | Operational reading |
|----------|---------------------|
| **P50** | Central schedule: the MW you would bid or nominate as the expected plant output. |
| **P10** | Downside / firm capacity: output you can count on ~90% of hours. Useful for reserve **need** (how much extra MW the rest of the fleet must cover if solar under-delivers). |
| **P90** | Upside / congestion / curtailment risk: output that shows up on high-resource hours. Useful for ramp and export constraints. |

The P10–P90 band is an 80% prediction interval. If empirical coverage on rolling-origin folds is well below 80%, a dispatcher using the band as a reserve envelope is **under-hedged**. If coverage is far above 80%, the band is too wide and you over-procure reserve. CRPS and pinball (not MAE) are the metrics that decide whether a training run is better for this use.

Serving exposes the same three quantiles — never a single MW point without the band.

## Inference mode: batch (day-ahead)

Day-ahead TFT needs a **168h encoder**, known-future calendar/solar rows for all **24 decoder steps**, and a rebuilt `TimeSeriesDataSet`. That work belongs on a **schedule** (e.g. nightly precompute), not inside a request handler.

| Mode | When it fits | This repo |
|------|----------------|-----------|
| **Batch** | Day-ahead dispatch: forecasts fixed until the next scheduled run | **Default** — API serves cached quantiles from MLflow Model Registry |
| **On-demand** | Intra-day reforecasts when new NWP or meter data arrives | Not implemented; `known_future` on the request schema is reserved for a later path |
| **Streaming** | Sub-hourly updates pushed to subscribers | Out of scope; would matter for real-time reserve or ramp alerts |

The API resolves `models:/tft-solar-quantile/Production` via `get_production_model()` — **not** a local checkpoint path.

### Register + serve locally

```powershell
pip install -e ".[serve]"
python scripts/register_model.py
uvicorn energy_forecasting.api.app:app --host 127.0.0.1 --port 8000
```

Smoke the endpoints:

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/plants
curl http://127.0.0.1:8000/metrics
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" -d "{\"plant_id\":\"DE_PV_001\",\"horizon\":24}"
```

**Request** (`POST /forecast`):

```json
{
  "plant_id": "DE_PV_001",
  "horizon": 24,
  "as_of": null,
  "known_future": null
}
```

**Response** (excerpt):

```json
{
  "quantiles": [0.1, 0.5, 0.9],
  "inference_mode": "batch",
  "model_name": "tft-solar-quantile",
  "model_version": "1",
  "model_stage": "Production",
  "forecasts": [
    {
      "timestamp": "2020-01-30T10:00:00Z",
      "plant_id": "DE_PV_001",
      "horizon": 1,
      "pred_q10": 37.1,
      "pred_q50": 212.0,
      "pred_q90": 462.9,
      "actual": 205.0
    }
  ]
}
```

First request loads the pyfunc model from the registry (~seconds); warm lookups are sub-second. Quantiles are clipped to enforce P10 ≤ P50 ≤ P90.

## Docker

Multi-stage **serve-only** image (~**1.39 GB** — no torch). Model artifacts are **pulled at runtime** from MLflow, not baked in. See [docker/README.md](docker/README.md) for bake-vs-pull trade-offs and CPU-torch trim notes.

```powershell
docker build -t energy-forecasting-api .
docker compose up --build   # API :8000, MLflow :5000, dashboard :8501
```

Seed a smoke Production model into the compose volume, then open the dashboard on :8501 or hit `POST /forecast` — details in [docker/README.md](docker/README.md).

### Dashboard

```powershell
pip install -r dashboard/requirements.txt
$env:API_URL = "http://127.0.0.1:8000"
streamlit run dashboard/app.py
```

Site selector, horizon slider, P50 line with P10–P90 band over actuals, and a pinball/coverage panel. See [dashboard/README.md](dashboard/README.md).

## Experiment tracking

**MLflow is the backbone** (self-managed, `./mlruns`). Each training run logs:

- **Params:** horizon, hidden size, quantiles, encoder length, lr, dropout, batch size, folds
- **Metrics per fold:** pinball_q10 / q50 / q90, CRPS, PI coverage
- **Artifacts:** metrics CSV, OOF parquet (`pred_q10/50/90`), quantile-band plot, Lightning checkpoint

**W&B is a taste, not the source of truth.** Mirror one run:

```powershell
pip install -e ".[wandb]"
# set WANDB_API_KEY, then:
python scripts/train.py --n-splits 2 --max-epochs 2 --max-plants 2 --hidden-size 16 --limit-train-batches 20 --limit-val-batches 5 --wandb --run-name tft-h16-wandb
```

| | MLflow (here) | Weights & Biases |
|--|---------------|------------------|
| Hosting | Local directory or self-hosted server | Hosted SaaS (auth + cloud artifact store) |
| Cost / lock-in | Open source; you own the `mlruns` folder | Convenient UI; data lives on their platform |
| This repo | Default tracking URI `./mlruns` | Optional `--wandb` flag on `scripts/train.py` |

## Layout

```
src/energy_forecasting/data/    # OPSD + Open-Meteo, features
src/energy_forecasting/model/   # TFT, metrics, CV, MLflow tracking
src/energy_forecasting/api/     # P10/P50/P90 schema + FastAPI batch inference
src/energy_forecasting/model/registry.py  # MLflow Model Registry + get_production_model()
scripts/train.py                # rolling-origin CV + tracking
scripts/register_model.py       # register best run → Production
scripts/fetch_data.py
tests/                          # unit tests on synthetic data (CI)
dashboard/                      # Streamlit dashboard (:8501 in compose)
docker/                         # Dockerfile, compose notes, seed helper
docker-compose.yml              # API + MLflow + dashboard
.github/workflows/ci.yml
```

## Docs

| Doc | Purpose |
|-----|---------|
| [AGENT.md](AGENT.md) | Rules for agents working in this repo |
| [handover.md](handover.md) | Live status, API, next step |
| [WEEK_09_REFLECTION.md](WEEK_09_REFLECTION.md) | Week 9 notes |
| [data/README.md](data/README.md) | Sources and schemas |
| [notebooks/README.md](notebooks/README.md) | Thin drivers only |

## License

Apache-2.0. See [LICENSE](LICENSE).

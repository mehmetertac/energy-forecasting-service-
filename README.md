# energy-forecasting-service

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

Serving (FastAPI + Streamlit, later this week) will expose the same three quantiles — never a single MW point without the band.

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
src/energy_forecasting/api/     # P10/P50/P90 schema; FastAPI stub
scripts/train.py                # rolling-origin CV + tracking
scripts/fetch_data.py
tests/                          # unit tests on synthetic data (CI)
dashboard/                      # Streamlit (next)
docker/                         # image (next)
.github/workflows/tests.yml
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

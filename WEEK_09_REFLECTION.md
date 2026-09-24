# WEEK_09_REFLECTION

Week 9 — MLOps Essentials. Goal: ship, don't just notebook.

## 2026-09-19 — Repo restructure + MLflow

- Ported Week 5 TFT + feature pipeline from `multi-site-solar-hierarchy` into `src/energy_forecasting/` (`data/`, `model/`, `api/`).
- Notebooks stay thin; training is `scripts/train.py`.
- Left MinT / N-HiTS / attention plots in Week 5. This repo needs a serving-shaped TFT, not the full hierarchy paper.
- MLflow is the backbone (`./mlruns`). W&B is an optional `--wandb` mirror so we can speak to hosted vs self-managed tracking.
- CI runs unit tests without torch/OPSD so every push is cheap and still meaningful.
- Prerequisite: `fetch_data.py` rebuilt 12 plants × 17,539 hours; plant-sum vs region max diff 1.09e-11 MW.
- Logged three CPU smoke runs (`hidden_size` 16/32/64) into `./mlruns` with pinball, CRPS, PI coverage, band plot, and checkpoints.
- Horizon-1 merge on those smoke folds only hit 2 rows — fine for proving the tracker, not for quoting coverage.
- W&B path is `--wandb`; no API key in this environment so the live mirror is still outstanding.

## 2026-09-20 — Model Registry + FastAPI

- Registered best smoke run (`tft-h16-smoke`, lowest `mean_crps`) as `tft-solar-quantile` v1 → Staging → Production.
- `get_production_model()` resolves `models:/tft-solar-quantile/Production`; the API never reads checkpoint paths directly.
- Serving format: **MLflow pyfunc** wrapping a cached forecast parquet (+ `.ckpt` stored as artifact for a future on-demand path). No ONNX yet.
- **Inference mode: batch.** Day-ahead precompute on a schedule; `POST /forecast` serves cached P10/P50/P90 with model version in the payload. On-demand / streaming reserved for intra-day reforecasts.
- Local smoke: uvicorn + `/health` + `/forecast`; quantile ordering enforced; warm latency sub-second after first load.

## Open questions

- How small can a honest smoke TFT be on CPU and still produce coverage numbers that are not noise?

## 2026-09-24 — Docker + Streamlit dashboard (Week 9 close-out)

### What did I build?

- **Importable package** — Week 5 TFT + feature pipeline under `src/energy_forecasting/`; training is `scripts/train.py`, not notebooks.
- **MLflow backbone** — params, per-fold pinball/CRPS/coverage, OOF parquet, band plots, checkpoints in `./mlruns`.
- **Model Registry** — `tft-solar-quantile` registered from best smoke run (`mean_crps`), promoted Staging → Production via `scripts/register_model.py`.
- **Batch FastAPI** — `GET /health`, `GET /plants`, `GET /metrics`, `POST /forecast` serving cached P10/P50/P90 from an MLflow pyfunc (no torch at request time). Optional `actual` MW when the cached parquet has `y` / `power_mw`.
- **Docker** — multi-stage serve image (~1.39 GB, no torch); compose brings up API (:8000), MLflow (:5000), and Streamlit dashboard (:8501) with one `docker compose up --build`.
- **Streamlit dashboard** — site selector, horizon slider, headline Altair chart (P50 + P10–P90 band over actuals), pinball/coverage panel, model version from the served payload.
- **CI** — ruff, pytest (synthetic data, no torch/OPSD), Docker build on every push/PR.

### What's still fuzzy?

- **Registry staging flows** — `register_best_model` transitions Staging → Production in one script call. There is no human review gate, canary, or rollback story beyond MLflow version history.
- **Image size** — serve image is ~1.39 GB without torch (pyarrow, scipy, matplotlib, MLflow stack). Adding on-demand TFT inference with CPU torch (~200 MB wheel) or default CUDA torch (~2 GB) would change the trade-off.
- **Test coverage gaps** — CI covers schema, quantile ordering, registry smoke, and dashboard client helpers. It does not run a real TFT fold, W&B mirror, or compose end-to-end. Horizon-1 smoke coverage had only 2 rows — fine for proving the tracker, not for quoting reserve envelopes.
- **W&B** — `--wandb` is wired; no live mirror run without `WANDB_API_KEY`.

### How would a real utility forecasting desk consume this?

| Role | Surface | Why |
|------|---------|-----|
| **Dispatch / forecast analyst** | Streamlit dashboard (:8501) | Visual check before day-ahead gate: band width, recent pinball/coverage, which model version is live. |
| **Nomination / EMS / reserve systems** | `POST /forecast` API (:8000) | Machine-readable P10/P50/P90 + `model_version` for schedules, reserve need, and audit trails. |
| **Modeling / MLOps** | MLflow UI (:5000), `scripts/train.py`, `scripts/register_model.py` | Train, compare runs, promote Production. The desk reads the result; it does not run promotion. |

The dashboard and API expose the **same contract** — probabilistic bands, not a collapsed point forecast.

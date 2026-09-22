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

## Later this week

Docker image, Streamlit bands, `WEEK_09_REFLECTION` wrap-up.

# From notebook to forecast service: shipping a probabilistic TFT with MLflow, Docker, FastAPI, and CI

**Paste-ready LinkedIn post**

---

Most served forecasts collapse to a single MW number. The uncertainty band gets dropped somewhere between training and production.

I shipped a probabilistic solar forecasting service where **P10 / P50 / P90 survive the whole stack**:

**Training** — Temporal Fusion Transformer with quantile loss (0.1 / 0.5 / 0.9), rolling-origin CV, pinball + CRPS + PI coverage logged to MLflow.

**Registry** — Best run promoted to `models:/tft-solar-quantile/Production`. Promotion does not require rebuilding the Docker image.

**Batch job** — Nightly `daily_forecast.py` loads the Production pyfunc, slices day-ahead origins, and publishes to SQLite. Day-ahead TFT needs a 168h encoder and 24 known-future steps — that work belongs on a schedule, not in a request handler.

**FastAPI** — `POST /forecast` serves cached quantiles in sub-second time. Same three numbers the model trained on, with model version in every payload.

**Streamlit dashboard** — P50 line with P10–P90 band over actuals, pinball/coverage panel. The dispatch analyst and the EMS read the same contract.

**CI** — ruff, pytest (no torch in CI), Docker build on every push.

Why it matters for dispatch: P50 is the nomination, P10 is firm capacity / reserve need, P90 is upside / curtailment risk. If the P10–P90 band's empirical coverage drifts off ~80%, you're under-hedged on reserve — CRPS and pinball decide retrain, not MAE.

Repo: https://github.com/mehmetertac/energy-forecasting-service-

`docker compose up --build` → dashboard on :8501, API on :8000, MLflow on :5000.

#MLOps #TimeSeries #EnergyForecasting #FastAPI #MLflow

---

**Shorter variant (280 chars for a teaser + link):**

Shipped a probabilistic solar TFT where P10/P50/P90 survive training → MLflow registry → batch job → FastAPI → Streamlit. Most served forecasts drop the band. `docker compose up --build` to try it.

https://github.com/mehmetertac/energy-forecasting-service-

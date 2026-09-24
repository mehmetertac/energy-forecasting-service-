# Dashboard

Streamlit demo for Week 9 — reads the FastAPI service, never the registry directly.

## Run locally

```powershell
pip install -r dashboard/requirements.txt
# API must be running on :8000 with a Production model registered
$env:API_URL = "http://127.0.0.1:8000"
streamlit run dashboard/app.py
```

Open http://127.0.0.1:8501

## Compose

`docker compose up --build` starts the dashboard on **:8501** with `API_URL=http://api:8000`.

## Features

- **Site selector** — `GET /plants`
- **Horizon slider** — 1–24 h, passed to `POST /forecast`
- **Headline chart** — P50 line, shaded P10–P90 band, actual MW when cached
- **Metrics panel** — pinball q50, PI coverage from `GET /metrics`, model version from forecast payload

The chart keeps only the **latest forecast origin** so a full OOF table does not plot every historical day.

## Layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI + Altair chart |
| `client.py` | HTTP client, latest-origin filter |
| `Dockerfile` | Lean image (streamlit, httpx, pandas, altair) |
| `requirements.txt` | Dashboard-only deps |

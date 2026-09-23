# Docker

Multi-stage **serve-only** image for batch inference (`GET /health`, `POST /forecast`). The production model is **not baked in** — it is loaded from MLflow Model Registry at runtime.

## Image size (measured 2026-09-22)

| Image | Size | Notes |
|-------|------|-------|
| `energy-forecasting-api:latest` | **~1.39 GB** | `[serve]` extra only — no torch |

Largest layers: **pyarrow** (~40 MB wheel), **scipy**, **matplotlib**, **mlflow**, **pandas**. This is lean for an MLflow stack but still heavy because MLflow pulls Flask/SQLAlchemy/scientific deps transitively.

### What we trimmed (and what to trim next)

**Already omitted from this image:**

- `train` extra — no **torch**, **pytorch-lightning**, or **pytorch-forecasting**
- `dev`, `wandb`, `notebooks` extras
- Model artifacts (`mlruns/`, checkpoints) — pulled at runtime
- Compiler toolchain in the final stage (multi-stage build)

**If you add on-demand TFT inference later**, torch dominates image size:

| Torch install | Approx. Linux wheel size |
|---------------|--------------------------|
| Default PyPI (CUDA build) | **~2 GB** |
| CPU index (`https://download.pytorch.org/whl/cpu`) | **~200 MB** |

Use the CPU index for serve images that only run inference on CPU. Keep batch parquet serving (current path) to avoid torch entirely.

## Model artifacts: bake vs pull (we chose pull)

| Approach | Pros | Cons |
|----------|------|------|
| **Bake into image** | `docker run` works offline; no volume wiring | Every Production promotion requires rebuild; `mlruns/` + checkpoints bloat layers; store is gitignored so image is not reproducible from git alone |
| **Pull at runtime (chosen)** | Image stays code-only; registry updates without rebuild; matches MLOps workflow | Needs MLflow tracking URI + artifact access; empty store → `GET /health` unhealthy, `POST /forecast` 503 |

Compose runs an MLflow server; the API uses `MLFLOW_TRACKING_URI=http://mlflow:5000` and shares the `mlflow-data` volume for artifact paths. Standalone `docker run` can mount a file store at `/mlflow`.

## Build

```powershell
docker build -t energy-forecasting-api .
```

Lockfile (`requirements.lock`) is pinned for Linux / Python 3.11. Regenerate inside slim (avoids Windows-only deps like `pywin32`):

```powershell
docker run --rm -v ${PWD}:/app -w /app python:3.11-slim-bookworm bash -c `
  "pip install -q '.[serve]' && pip freeze | grep -vi '^energy-forecasting-service' > requirements.lock"
```

## Run (standalone API)

Seed a file store (must use **`/mlflow`** as the store path so artifact URIs match the mount):

```powershell
docker run --rm `
  -v ${PWD}:/app `
  -v ${PWD}/docker/mlruns-seed-linux:/mlflow `
  -w /app python:3.11-slim-bookworm `
  bash -c "pip install -q '.[serve]' && python scripts/seed_docker_mlruns.py --store /mlflow"
```

Run the API (non-root `uid 1000`; seed script chmods artifacts for the app user):

```powershell
docker run --rm -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=file:///mlflow `
  -v ${PWD}/docker/mlruns-seed-linux:/mlflow `
  energy-forecasting-api
```

Smoke from the host:

```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/forecast `
  -H "Content-Type: application/json" `
  -d '{\"plant_id\":\"DE_PV_001\",\"horizon\":24}'
```

After real training, mount your host `./mlruns` instead (register with `scripts/register_model.py` first).

## Compose (API + MLflow + dashboard stub)

```powershell
docker compose up --build
```

| Service | Port | Role |
|---------|------|------|
| `api` | 8000 | FastAPI batch inference |
| `mlflow` | 5000 | Tracking + registry + artifact store |
| `dashboard` | 8501 | Streamlit placeholder (stdlib HTTP stub) |

Seed Production model into the compose volume (once per fresh volume):

```powershell
docker run --rm `
  -v energy-forecasting-service_mlflow-data:/mlflow `
  -v ${PWD}:/app -w /app python:3.11-slim-bookworm `
  bash -c "pip install -q '.[serve]' && python scripts/seed_docker_mlruns.py --store /mlflow"
```

Or register a real run after training:

```powershell
# host: python scripts/register_model.py  (writes to ./mlruns)
# then copy or mount ./mlruns into the mlflow volume
```

Verified: `POST /forecast` returns 24 P10/P50/P90 rows for `DE_PV_001` when the volume holds `tft-solar-quantile` Production.

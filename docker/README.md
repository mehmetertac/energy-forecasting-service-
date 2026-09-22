# Docker

FastAPI batch inference is wired (`POST /forecast`, `GET /health`) and loads the production model from MLflow Model Registry via `get_production_model()`.

The Docker image (FastAPI + `[serve]` extra + registry URI) is the next packaging step — not built yet.

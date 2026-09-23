# Multi-stage serve image — batch inference only (no torch / train extra).
# Regenerate lockfile: pip install ".[serve]" && pip freeze > requirements.lock

FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.lock .
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock

FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home app

COPY --from=builder /install /usr/local
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --no-deps . && \
    rm -rf /root/.cache

USER app

EXPOSE 8000

CMD ["uvicorn", "energy_forecasting.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

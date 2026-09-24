"""HTTP client for the energy-forecasting FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

DEFAULT_API_URL = "http://127.0.0.1:8000"


def api_url() -> str:
    return os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")


def fetch_health() -> dict[str, Any]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{api_url()}/health")
        response.raise_for_status()
        return response.json()


def fetch_plants() -> list[str]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{api_url()}/plants")
        response.raise_for_status()
        return response.json()["plants"]


def fetch_metrics() -> dict[str, Any]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{api_url()}/metrics")
        response.raise_for_status()
        return response.json()


def fetch_forecast(*, plant_id: str, horizon: int) -> dict[str, Any]:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{api_url()}/forecast",
            json={"plant_id": plant_id, "horizon": horizon},
        )
        response.raise_for_status()
        return response.json()


def filter_latest_origin(forecasts: list[dict[str, Any]], *, horizon: int) -> pd.DataFrame:
    """Keep rows from the latest forecast origin with horizons <= ``horizon``."""
    if not forecasts:
        return pd.DataFrame()

    df = pd.DataFrame(forecasts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["origin"] = df["timestamp"] - pd.to_timedelta(df["horizon"], unit="h")
    latest_origin = df["origin"].max()
    work = df[(df["origin"] == latest_origin) & (df["horizon"] <= horizon)].copy()
    return work.sort_values("horizon").reset_index(drop=True)

"""Fetch hourly weather from Open-Meteo Archive API."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from energy_forecasting.config import (
    DATA_END,
    DATA_START,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_REQUEST_DELAY_S,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

WEATHER_CACHE_DIR = RAW_DIR / "open_meteo"
HOURLY_VARS = "temperature_2m,shortwave_radiation,cloud_cover"


def _cache_path(plant_id: str) -> Path:
    return WEATHER_CACHE_DIR / f"{plant_id}.json"


def fetch_plant_weather(
    plant_id: str,
    lat: float,
    lon: float,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch hourly weather for one plant; cache JSON on disk."""
    cache_file = _cache_path(plant_id)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": DATA_START,
            "end_date": DATA_END,
            "hourly": HOURLY_VARS,
            "timezone": "UTC",
        }
        logger.info("Fetching Open-Meteo weather for %s (%.4f, %.4f)", plant_id, lat, lon)
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(OPEN_METEO_REQUEST_DELAY_S)

    hourly = payload["hourly"]
    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"], utc=True),
            "temp_c": hourly["temperature_2m"],
            "ghi_wm2": hourly["shortwave_radiation"],
            "cloud_cover_pct": hourly["cloud_cover"],
        }
    )
    weather["plant_id"] = plant_id
    return weather


def fetch_all_plant_weather(
    plants: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch weather for all plants and return long-format DataFrame."""
    frames = [
        fetch_plant_weather(
            plant_id=row["plant_id"],
            lat=row["lat"],
            lon=row["lon"],
            force=force,
        )
        for _, row in plants.iterrows()
    ]
    return pd.concat(frames, ignore_index=True)

"""Download and parse Open Power System Data (OPSD) sources."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from energy_forecasting.config import (
    DATA_END,
    DATA_START,
    OPSD_RENEWABLE_PLANTS_URL,
    OPSD_TIME_SERIES_URL,
    RAW_DIR,
    REGIONAL_SOLAR_COLUMNS,
)

logger = logging.getLogger(__name__)

TIME_SERIES_CACHE = RAW_DIR / "opsd_time_series_60min.csv"
PLANTS_CACHE = RAW_DIR / "opsd_renewable_plants_de.csv"


def _download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.info("Using cached file: %s", destination)
        return destination

    logger.info("Downloading %s -> %s", url, destination)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def fetch_time_series(force: bool = False) -> Path:
    return _download_file(OPSD_TIME_SERIES_URL, TIME_SERIES_CACHE, force=force)


def fetch_renewable_plants(force: bool = False) -> Path:
    return _download_file(OPSD_RENEWABLE_PLANTS_URL, PLANTS_CACHE, force=force)


def _resolve_solar_column(columns: pd.Index) -> str:
    for name in REGIONAL_SOLAR_COLUMNS:
        if name in columns:
            return name
    raise ValueError(
        "No regional solar column found. Expected one of: "
        f"{REGIONAL_SOLAR_COLUMNS}. Available columns sample: {list(columns[:20])}"
    )


def load_regional_solar(path: Path | None = None) -> pd.DataFrame:
    """Return hourly regional solar generation in MW."""
    csv_path = path or TIME_SERIES_CACHE
    if not csv_path.exists():
        fetch_time_series()

    df = pd.read_csv(csv_path, parse_dates=["utc_timestamp"])
    solar_col = _resolve_solar_column(df.columns)

    region = (
        df[["utc_timestamp", solar_col]]
        .rename(columns={"utc_timestamp": "timestamp", solar_col: "power_mw"})
        .dropna(subset=["power_mw"])
    )
    region["timestamp"] = pd.to_datetime(region["timestamp"], utc=True)
    start = pd.Timestamp(DATA_START, tz="UTC")
    end = pd.Timestamp(f"{DATA_END} 23:00:00", tz="UTC")
    region = region[(region["timestamp"] >= start) & (region["timestamp"] <= end)]
    region = region.sort_values("timestamp").reset_index(drop=True)
    if region.empty:
        raise ValueError(
            f"No regional solar data in range {DATA_START} to {DATA_END}. "
            "OPSD time_series package 2020-10-06 covers through 2020-09-30."
        )
    return region


def load_solar_plants(path: Path | None = None) -> pd.DataFrame:
    """Return DE solar plants from OPSD renewable registry."""
    csv_path = path or PLANTS_CACHE
    if not csv_path.exists():
        fetch_renewable_plants()

    plants = pd.read_csv(csv_path, low_memory=False)
    solar_mask = plants["energy_source_level_2"].str.contains("Solar", case=False, na=False)
    solar = plants.loc[solar_mask].copy()

    rename_map = {
        "electrical_capacity": "capacity_mw",
        "lon": "lon",
        "lat": "lat",
        "name": "name",
        "municipality": "municipality",
        "federal_state": "federal_state",
    }
    solar = solar.rename(columns=rename_map)
    required = ["capacity_mw", "lat", "lon"]
    solar = solar.dropna(subset=required)
    solar["capacity_mw"] = pd.to_numeric(solar["capacity_mw"], errors="coerce")
    solar = solar.dropna(subset=["capacity_mw"])
    solar = solar[solar["capacity_mw"] > 0]
    return solar.reset_index(drop=True)

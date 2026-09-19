"""Load processed dataset artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.config import PROCESSED_DIR


def load_plants_metadata(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or PROCESSED_DIR / "plants_metadata.parquet")


def load_plants_hourly(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or PROCESSED_DIR / "plants_hourly.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_region_hourly(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or PROCESSED_DIR / "region_hourly.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

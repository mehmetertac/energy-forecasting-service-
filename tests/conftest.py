"""Shared synthetic fixtures (no network, no processed parquet)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_hourly() -> pd.DataFrame:
    times = pd.date_range("2019-06-01", periods=96, freq="h", tz="UTC")
    rows = []
    for plant_id, lat_off in (("DE_PV_001", 0.0), ("DE_PV_002", 1.0)):
        hour = times.hour.to_numpy()
        ghi = np.clip(800 * np.sin(np.pi * (hour - 6) / 12), 0, None)
        power = ghi / 800.0 * (20.0 + lat_off)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": times,
                    "plant_id": plant_id,
                    "power_mw": power,
                    "capacity_mw": 25.0 + lat_off,
                    "ghi_wm2": ghi,
                    "temp_c": 18.0 + lat_off,
                    "cloud_cover_pct": 20.0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": ["DE_PV_001", "DE_PV_002"],
            "name": ["Plant A", "Plant B"],
            "capacity_mw": [25.0, 26.0],
            "lat": [51.0, 52.0],
            "lon": [10.0, 11.0],
            "region_id": ["DE_LU_PV", "DE_LU_PV"],
            "federal_state": ["Niedersachsen", "Bayern"],
            "subregion_id": ["DE_SR_NIEDERSACHSEN", "DE_SR_BAYERN"],
        }
    )


@pytest.fixture
def prepared_tft_frame() -> pd.DataFrame:
    """Hand-built TFT-like frame so role tests do not call pvlib."""
    times = pd.date_range("2019-06-01", periods=72, freq="h", tz="UTC")
    frames = []
    for plant_id in ("DE_PV_001", "DE_PV_002"):
        hour = times.hour.to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": times,
                    "plant_id": plant_id,
                    "unique_id": plant_id,
                    "subregion_id": "DE_SR_TEST",
                    "power_mw": np.clip(10 * np.sin(np.pi * (hour - 6) / 12), 0, None),
                    "capacity_mw": 20.0,
                    "lat": 51.0,
                    "lon": 10.0,
                    "hour": hour,
                    "day_of_week": times.dayofweek,
                    "month": times.month,
                    "week_of_year": times.isocalendar().week.astype(int),
                    "is_weekend": (times.dayofweek >= 5).astype(int),
                    "is_holiday": 0,
                    "days_since_last_holiday": 3,
                    "days_to_next_holiday": 4,
                    "sin_hour": np.sin(2 * np.pi * hour / 24),
                    "cos_hour": np.cos(2 * np.pi * hour / 24),
                    "sin_day_of_year": 0.1,
                    "cos_day_of_year": 0.9,
                    "ghi_wm2": np.clip(700 * np.sin(np.pi * (hour - 6) / 12), 0, None),
                    "temp_c": 18.0,
                    "cloud_cover_pct": 15.0,
                    "clearsky_index": 0.8,
                    "temp_c_sq": 324.0,
                    "cloud_frac": 0.15,
                    "expected_cf": 0.8,
                    "sun_apparent_zenith": 40.0,
                    "cs_ghi": 800.0,
                    "time_idx": np.arange(len(times)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)

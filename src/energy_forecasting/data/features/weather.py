"""Weather-derived features from Open-Meteo columns (NWP proxy)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived weather/NWP proxy features from base Open-Meteo columns."""
    out = df.copy()
    if "ghi_wm2" in out.columns and "cs_ghi" in out.columns:
        out["clearsky_index"] = np.divide(
            out["ghi_wm2"],
            out["cs_ghi"].replace(0, np.nan),
        ).clip(0.0, 1.5).fillna(0.0)
    elif "ghi_wm2" in out.columns:
        out["clearsky_index"] = np.nan

    if "temp_c" in out.columns:
        out["temp_c_sq"] = out["temp_c"] ** 2

    if "cloud_cover_pct" in out.columns:
        out["cloud_frac"] = out["cloud_cover_pct"] / 100.0

    if "capacity_mw" in out.columns and "clearsky_index" in out.columns:
        out["expected_cf"] = out["clearsky_index"].clip(0.0, 1.0)

    return out

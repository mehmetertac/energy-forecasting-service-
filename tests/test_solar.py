"""Solar feature tests (pvlib, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.data.features.solar import add_solar_features

BERLIN_LAT = 52.52
BERLIN_LON = 13.405


def _solar_frame_at(local_time: str) -> pd.DataFrame:
    ts = pd.DatetimeIndex([pd.Timestamp(local_time, tz="Europe/Berlin")])
    base = pd.DataFrame({"power_mw": [0.0]}, index=ts)
    return add_solar_features(base, BERLIN_LAT, BERLIN_LON, tz="Europe/Berlin")


def test_solar_noon_high_elevation_and_ghi():
    """Summer solstice local noon: sun above horizon, clear-sky GHI well above zero."""
    out = _solar_frame_at("2020-06-21 12:00")
    assert out["sun_apparent_elevation"].iloc[0] > 50.0
    assert out["cs_ghi"].iloc[0] > 500.0
    for col in ("sun_apparent_zenith", "sun_apparent_elevation", "sun_azimuth", "cs_ghi"):
        assert np.isfinite(out[col].iloc[0])


def test_solar_midnight_low_elevation_and_ghi():
    """Local midnight: sun below horizon, clear-sky GHI near zero."""
    out = _solar_frame_at("2020-06-21 00:00")
    assert out["sun_apparent_elevation"].iloc[0] < 0.0
    assert out["cs_ghi"].iloc[0] == pytest.approx(0.0, abs=1.0)
    for col in ("sun_apparent_zenith", "sun_apparent_elevation", "sun_azimuth", "cs_ghi"):
        assert np.isfinite(out[col].iloc[0])

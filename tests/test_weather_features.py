"""Weather-derived feature tests."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.data.features.weather import add_weather_features


def test_weather_features_derived():
    df = pd.DataFrame(
        {
            "ghi_wm2": [400.0, 0.0],
            "cs_ghi": [800.0, 0.0],
            "temp_c": [10.0, 2.0],
            "cloud_cover_pct": [50.0, 0.0],
            "capacity_mw": [20.0, 20.0],
        }
    )
    out = add_weather_features(df)
    assert out["clearsky_index"].iloc[0] == pytest.approx(0.5)
    assert out["cloud_frac"].iloc[0] == pytest.approx(0.5)
    assert out["temp_c_sq"].iloc[0] == pytest.approx(100.0)
    assert out["expected_cf"].iloc[0] == pytest.approx(0.5)

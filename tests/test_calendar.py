"""Calendar feature tests (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.data.features.calendar import make_calendar_features


def test_calendar_columns_and_weekend():
    idx = pd.date_range("2024-01-06", periods=48, freq="h", tz="Europe/Berlin")  # Saturday
    cal = make_calendar_features(idx, country="DE")
    expected = {
        "hour",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "is_holiday",
        "days_since_last_holiday",
        "days_to_next_holiday",
        "sin_hour",
        "cos_hour",
        "sin_day_of_year",
        "cos_day_of_year",
    }
    assert expected.issubset(cal.columns)
    assert cal["is_weekend"].iloc[0] == 1
    assert cal["hour"].iloc[0] == 0
    assert len(cal) == 48


def test_calendar_rejects_non_index():
    with pytest.raises(TypeError):
        make_calendar_features(["2024-01-01"])  # type: ignore[arg-type]

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


def test_calendar_new_year_expected_values():
    """2024-01-01 00:00 Europe/Berlin: Monday, January, New Year, not weekend."""
    idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="Europe/Berlin")
    cal = make_calendar_features(idx, country="DE")
    row = cal.iloc[0]
    assert row["hour"] == 0
    assert row["day_of_week"] == 0  # Monday
    assert row["month"] == 1
    assert row["is_weekend"] == 0
    assert row["is_holiday"] == 1
    assert row["sin_hour"] == pytest.approx(0.0, abs=1e-6)
    assert row["cos_hour"] == pytest.approx(1.0, abs=1e-6)


def test_calendar_cyclical_hour_six():
    idx = pd.date_range("2024-01-01 06:00", periods=1, freq="h", tz="Europe/Berlin")
    cal = make_calendar_features(idx, country="DE")
    row = cal.iloc[0]
    assert row["hour"] == 6
    assert row["sin_hour"] == pytest.approx(1.0, abs=1e-6)
    assert row["cos_hour"] == pytest.approx(0.0, abs=1e-6)


def test_calendar_rejects_non_index():
    with pytest.raises(TypeError):
        make_calendar_features(["2024-01-01"])  # type: ignore[arg-type]

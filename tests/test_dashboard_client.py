"""Dashboard HTTP client helpers (no Streamlit import)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_dashboard = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(_dashboard))

from client import filter_latest_origin  # noqa: E402


def _forecast_rows(*, plant_id: str = "DE_PV_001") -> list[dict]:
    origin = pd.Timestamp("2020-06-01T00:00:00Z")
    older_origin = pd.Timestamp("2020-05-31T00:00:00Z")
    rows: list[dict] = []
    for base in (older_origin, origin):
        for h in range(1, 25):
            ts = base + pd.Timedelta(hours=h)
            rows.append(
                {
                    "timestamp": ts.isoformat().replace("+00:00", "Z"),
                    "plant_id": plant_id,
                    "horizon": h,
                    "pred_q10": float(h),
                    "pred_q50": float(h + 1),
                    "pred_q90": float(h + 2),
                    "actual": float(h + 1.5),
                }
            )
    return rows


def test_filter_latest_origin_keeps_newest_only():
    filtered = filter_latest_origin(_forecast_rows(), horizon=24)
    assert len(filtered) == 24
    origins = pd.to_datetime(filtered["timestamp"], utc=True) - pd.to_timedelta(filtered["horizon"], unit="h")
    assert origins.nunique() == 1
    assert origins.iloc[0] == pd.Timestamp("2020-06-01T00:00:00Z")


def test_filter_latest_origin_respects_horizon_slider():
    filtered = filter_latest_origin(_forecast_rows(), horizon=6)
    assert len(filtered) == 6
    assert filtered["horizon"].max() == 6


def test_filter_latest_origin_empty_input():
    assert filter_latest_origin([], horizon=24).empty

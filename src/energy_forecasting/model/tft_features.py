"""Feature preparation and role mapping for pytorch-forecasting TFT."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energy_forecasting.config import TIMESTAMP_COL
from energy_forecasting.data.features import build_plant_feature_frame, default_feature_columns

STATIC_REAL_COLS = ("capacity_mw", "lat", "lon")
STATIC_CATEGORICAL_COLS = ("plant_id", "subregion_id")

KNOWN_FUTURE_REAL_COLS = (
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
)

OBSERVED_UNKNOWN_REAL_COLS = (
    "ghi_wm2",
    "temp_c",
    "cloud_cover_pct",
    "clearsky_index",
    "temp_c_sq",
    "cloud_frac",
    "expected_cf",
)


@dataclass(frozen=True)
class TFTFeatureRoles:
    """Column groups for :class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`."""

    static_reals: tuple[str, ...]
    static_categoricals: tuple[str, ...]
    known_future_reals: tuple[str, ...]
    observed_unknown_reals: tuple[str, ...]


def _solar_known_cols(columns: pd.Index) -> tuple[str, ...]:
    return tuple(col for col in columns if col.startswith("sun_") or col.startswith("cs_"))


def resolve_feature_roles(frame: pd.DataFrame) -> TFTFeatureRoles:
    """Infer TFT feature roles from a prepared modeling frame."""
    cols = set(frame.columns)
    known = tuple(c for c in KNOWN_FUTURE_REAL_COLS if c in cols) + _solar_known_cols(frame.columns)
    observed = tuple(c for c in OBSERVED_UNKNOWN_REAL_COLS if c in cols)
    static_reals = tuple(c for c in STATIC_REAL_COLS if c in cols)
    static_cats = tuple(c for c in STATIC_CATEGORICAL_COLS if c in cols)
    return TFTFeatureRoles(
        static_reals=static_reals,
        static_categoricals=static_cats,
        known_future_reals=known,
        observed_unknown_reals=observed,
    )


def prepare_tft_frame(
    plants_hourly: pd.DataFrame,
    plants_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build TFT-ready long-format table with static fields, time_idx, and unique_id."""
    feat = build_plant_feature_frame(plants_hourly, plants_metadata)
    meta = plants_metadata.set_index("plant_id")
    static_cols = [c for c in STATIC_REAL_COLS if c in meta.columns]
    static = meta[static_cols].reset_index()
    if "capacity_mw" in feat.columns and "capacity_mw" in static.columns:
        feat = feat.drop(columns=["capacity_mw"])
    out = feat.merge(static, on="plant_id", how="left")

    out["plant_id"] = out["plant_id"].astype(str)
    out["unique_id"] = out["plant_id"]
    if "subregion_id" in out.columns:
        out["subregion_id"] = out["subregion_id"].astype(str)

    times = pd.Series(out[TIMESTAMP_COL].drop_duplicates().sort_values())
    time_idx_map = {ts: i for i, ts in enumerate(times)}
    out["time_idx"] = out[TIMESTAMP_COL].map(time_idx_map).astype(int)
    out = out.sort_values(["plant_id", "time_idx"]).reset_index(drop=True)
    return out


def default_tft_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return default exogenous columns for CV."""
    base = default_feature_columns(frame)
    extra = [c for c in STATIC_REAL_COLS if c in frame.columns and c not in base]
    return base + extra

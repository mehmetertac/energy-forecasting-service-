"""TFT feature-role mapping on synthetic frames (no torch)."""

from __future__ import annotations

from energy_forecasting.model.tft_features import (
    default_tft_feature_columns,
    prepare_tft_frame,
    resolve_feature_roles,
)


def test_feature_roles_nonempty(prepared_tft_frame):
    roles = resolve_feature_roles(prepared_tft_frame)
    assert roles.static_reals
    assert roles.static_categoricals
    assert roles.known_future_reals
    assert roles.observed_unknown_reals
    assert "ghi_wm2" in roles.observed_unknown_reals
    assert "hour" in roles.known_future_reals


def test_default_feature_columns_excludes_ids(prepared_tft_frame):
    cols = default_tft_feature_columns(prepared_tft_frame)
    assert "plant_id" not in cols
    assert "power_mw" not in cols
    assert "capacity_mw" in cols


def test_prepare_tft_frame_assigns_time_idx(synthetic_hourly, synthetic_metadata):
    frame = prepare_tft_frame(synthetic_hourly, synthetic_metadata)
    assert frame["unique_id"].nunique() == 2
    assert frame["time_idx"].min() == 0
    for _, group in frame.groupby("plant_id"):
        diffs = group["time_idx"].diff().dropna()
        assert (diffs == 1).all()
    for col in ("lat", "lon", "capacity_mw", "pred_placeholder"):
        if col == "pred_placeholder":
            continue
        assert col in frame.columns
        assert frame[col].notna().all()

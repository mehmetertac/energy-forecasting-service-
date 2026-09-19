"""Optional TimeSeriesDataSet / TFT tests (skipped without [train] extra)."""

from __future__ import annotations

import pytest


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except (OSError, ImportError):
        return False


pytestmark = [
    pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed"),
]


def test_timeseries_dataset_builds(prepared_tft_frame):
    pytest.importorskip("pytorch_forecasting")
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    from energy_forecasting.model.tft_features import resolve_feature_roles

    roles = resolve_feature_roles(prepared_tft_frame)
    max_idx = int(prepared_tft_frame["time_idx"].max()) - 24
    train_slice = prepared_tft_frame[prepared_tft_frame["time_idx"] <= max_idx].copy()

    dataset = TimeSeriesDataSet(
        train_slice,
        time_idx="time_idx",
        target="power_mw",
        group_ids=["plant_id"],
        max_encoder_length=24,
        max_prediction_length=12,
        min_encoder_length=12,
        static_categoricals=list(roles.static_categoricals),
        static_reals=list(roles.static_reals),
        time_varying_known_reals=list(roles.known_future_reals),
        time_varying_unknown_reals=list(roles.observed_unknown_reals),
        target_normalizer=GroupNormalizer(groups=["plant_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )
    assert len(dataset) > 0

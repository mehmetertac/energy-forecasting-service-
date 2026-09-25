"""Batch publish pipeline and SQLite forecast store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from energy_forecasting.model.registry import package_local_forecast
from energy_forecasting.serving.batch import (
    MissingCovariatesError,
    StaleModelError,
    publish_day_ahead_forecasts,
    slice_day_ahead_forecasts,
    validate_features_parquet,
    validate_forecast_frame,
)
from energy_forecasting.serving.store import (
    EmptyForecastError,
    StaleForecastError,
    get_batch_metadata,
    get_forecasts,
    list_plants,
    publish_batch,
)


def _sample_forecast_parquet(path: Path, *, with_actual: bool = False) -> None:
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    data = {
        "timestamp": ts,
        "plant_id": ["DE_PV_001"] * 24,
        "horizon": list(range(1, 25)),
        "pred_q10": [float(i) for i in range(24)],
        "pred_q50": [float(i + 1) for i in range(24)],
        "pred_q90": [float(i + 2) for i in range(24)],
    }
    if with_actual:
        data["y"] = [4.5] * 24
    pd.DataFrame(data).to_parquet(path, index=False)


def test_slice_day_ahead_keeps_latest_origin():
    ts = pd.date_range("2020-06-01", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": list(ts[:24]) + list(ts[24:]),
            "plant_id": ["DE_PV_001"] * 48,
            "horizon": list(range(1, 25)) * 2,
            "pred_q10": [1.0] * 48,
            "pred_q50": [2.0] * 48,
            "pred_q90": [3.0] * 48,
        }
    )
    sliced = slice_day_ahead_forecasts(df, max_horizon=24)
    assert len(sliced) == 24
    assert sliced["timestamp"].min() == ts[24]


def test_validate_forecast_frame_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_forecast_frame(pd.DataFrame())


def test_validate_features_parquet_missing_columns(tmp_path: Path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"hour": [1, 2]}).to_parquet(path)
    with pytest.raises(MissingCovariatesError, match="missing known-future"):
        validate_features_parquet(path)


def test_store_round_trip(tmp_path: Path):
    db_path = tmp_path / "forecasts.db"
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "plant_id": ["DE_PV_001"] * 24,
            "horizon": list(range(1, 25)),
            "pred_q10": [1.0] * 24,
            "pred_q50": [2.0] * 24,
            "pred_q90": [3.0] * 24,
        }
    )
    publish_batch(
        df,
        model_name="tft-test",
        model_version="1",
        model_stage="Production",
        run_id="run-1",
        db_path=db_path,
    )
    assert list_plants(db_path=db_path) == ["DE_PV_001"]
    meta, rows = get_forecasts(plant_id="DE_PV_001", horizon=24, db_path=db_path)
    assert meta.model_version == "1"
    assert len(rows) == 24


def test_stale_forecast_raises(tmp_path: Path):
    db_path = tmp_path / "forecasts.db"
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "plant_id": ["DE_PV_001"] * 24,
            "horizon": list(range(1, 25)),
            "pred_q10": [1.0] * 24,
            "pred_q50": [2.0] * 24,
            "pred_q90": [3.0] * 24,
        }
    )
    old = datetime.now(tz=UTC) - timedelta(hours=48)
    publish_batch(
        df,
        model_name="tft-test",
        model_version="1",
        model_stage="Production",
        run_id="run-1",
        generated_at=old,
        db_path=db_path,
    )
    with pytest.raises(StaleForecastError):
        get_batch_metadata(db_path=db_path, max_age_hours=36)


def test_empty_store_raises(tmp_path: Path):
    with pytest.raises(EmptyForecastError):
        get_batch_metadata(db_path=tmp_path / "missing.db")


def test_publish_day_ahead_from_registry(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    db_path = tmp_path / "forecasts.db"
    model_name = "tft-batch-test"

    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )

    meta = publish_day_ahead_forecasts(
        db_path=db_path,
        tracking_uri=tracking_uri,
        model_name=model_name,
    )

    assert meta.model_name == model_name
    _meta, rows = get_forecasts(plant_id="DE_PV_001", horizon=24, db_path=db_path)
    assert len(rows) == 24


def test_stale_model_refuses_publish(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    db_path = tmp_path / "forecasts.db"
    model_name = "tft-stale-model"

    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )

    with patch(
        "energy_forecasting.serving.batch.assert_model_fresh",
        side_effect=StaleModelError("stale model"),
    ):
        with pytest.raises(StaleModelError):
            publish_day_ahead_forecasts(
                db_path=db_path,
                tracking_uri=tracking_uri,
                model_name=model_name,
            )

    with pytest.raises(EmptyForecastError):
        get_batch_metadata(db_path=db_path)

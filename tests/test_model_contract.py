"""Serving-model contract: finite ordered quantiles and correct horizon length (no torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.model.registry import load_production_pyfunc, package_local_forecast


def _contract_forecast_parquet(path: Path, *, plant_id: str = "DE_PV_001") -> None:
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "plant_id": plant_id,
            "horizon": list(range(1, 25)),
            "pred_q10": [float(i) for i in range(24)],
            "pred_q50": [float(i + 1) for i in range(24)],
            "pred_q90": [float(i + 2) for i in range(24)],
        }
    )
    df.to_parquet(path, index=False)


@pytest.fixture
def contract_model(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _contract_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    model_name = "tft-contract-test"
    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    info, model = load_production_pyfunc(model_name=model_name, tracking_uri=tracking_uri)
    return info, model


def _assert_quantile_contract(rows: pd.DataFrame, expected_len: int) -> None:
    assert len(rows) == expected_len
    for col in ("pred_q10", "pred_q50", "pred_q90"):
        assert np.all(np.isfinite(rows[col].to_numpy(dtype=float)))
    assert (rows["pred_q10"] <= rows["pred_q50"]).all()
    assert (rows["pred_q50"] <= rows["pred_q90"]).all()


def test_model_contract_full_horizon(contract_model):
    _, model = contract_model
    rows = model.predict([{"plant_id": "DE_PV_001", "horizon": 24}])
    _assert_quantile_contract(rows, 24)
    assert set(rows["horizon"]) == set(range(1, 25))


def test_model_contract_short_horizon(contract_model):
    _, model = contract_model
    rows = model.predict([{"plant_id": "DE_PV_001", "horizon": 6}])
    _assert_quantile_contract(rows, 6)
    assert set(rows["horizon"]) == {1, 2, 3, 4, 5, 6}

"""MLflow Model Registry helpers (no torch)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy_forecasting.model.registry import (
    REGISTERED_MODEL_NAME,
    get_production_model,
    load_production_pyfunc,
    package_local_forecast,
    select_best_run,
)


def _sample_forecast_parquet(path: Path) -> None:
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "plant_id": "DE_PV_001",
            "horizon": list(range(1, 25)),
            "pred_q10": [float(i) for i in range(24)],
            "pred_q50": [float(i + 1) for i in range(24)],
            "pred_q90": [float(i + 2) for i in range(24)],
        }
    )
    df.to_parquet(path, index=False)


def test_package_and_get_production_model(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    model_name = f"{REGISTERED_MODEL_NAME}-test"

    result = package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
        run_name="unit-register",
    )
    assert result.stage == "Production"
    assert result.model_version == "1"

    info = get_production_model(model_name=model_name, tracking_uri=tracking_uri)
    assert info.version == "1"
    assert info.stage == "Production"
    assert info.uri == f"models:/{model_name}/Production"


def test_load_production_pyfunc_predict(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    model_name = f"{REGISTERED_MODEL_NAME}-pyfunc"

    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    info, model = load_production_pyfunc(model_name=model_name, tracking_uri=tracking_uri)
    assert info.version == "1"

    rows = model.predict([{"plant_id": "DE_PV_001", "horizon": 3}])
    assert len(rows) == 3
    assert set(rows["horizon"]) == {1, 2, 3}


def test_select_best_run(tmp_path: Path):
    import mlflow

    from energy_forecasting.model.tracking import _resolve_tracking_uri

    tracking_uri = tmp_path / "mlruns"
    uri = _resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("select-best-test")

    with mlflow.start_run(run_name="good"):
        mlflow.log_metric("mean_crps", 0.5)
    with mlflow.start_run(run_name="best"):
        mlflow.log_metric("mean_crps", 0.1)

    run_id = select_best_run(experiment_name="select-best-test", tracking_uri=tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["mean_crps"] == pytest.approx(0.1)


def test_select_best_run_missing_experiment(tmp_path: Path):
    with pytest.raises(RuntimeError, match="experiment not found"):
        select_best_run(experiment_name="missing-exp", tracking_uri=tmp_path / "mlruns")

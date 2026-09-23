"""FastAPI batch inference contract (no torch)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from energy_forecasting.api.app import create_app
from energy_forecasting.api.schema import enforce_quantile_order
from energy_forecasting.model.registry import ProductionModelInfo, package_local_forecast


def _sample_forecast_parquet(path: Path) -> None:
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "plant_id": ["DE_PV_001"] * 24,
            "horizon": list(range(1, 25)),
            "pred_q10": [3.0, 5.0, 1.0] + [float(i) for i in range(3, 24)],
            "pred_q50": [4.0] * 24,
            "pred_q90": [6.0] * 24,
        }
    )
    df.to_parquet(path, index=False)


def test_enforce_quantile_order():
    q10, q50, q90 = enforce_quantile_order(5.0, 3.0, 6.0)
    assert q10 <= q50 <= q90
    assert (q10, q50, q90) == (3.0, 5.0, 6.0)


def test_forecast_endpoint(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    model_name = "tft-api-test"

    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    info = ProductionModelInfo(
        name=model_name,
        version="1",
        stage="Production",
        run_id="test-run",
        uri=f"models:/{model_name}/Production",
    )

    def get_model():
        return info

    def load_pyfunc():
        from energy_forecasting.model.registry import load_production_pyfunc

        return load_production_pyfunc(model_name=model_name, tracking_uri=tracking_uri)

    client = TestClient(create_app(get_model=get_model, load_pyfunc=load_pyfunc))
    response = client.post("/forecast", json={"plant_id": "DE_PV_001", "horizon": 24})
    assert response.status_code == 200
    body = response.json()
    assert body["inference_mode"] == "batch"
    assert body["model_version"] == "1"
    assert body["model_stage"] == "Production"
    assert len(body["forecasts"]) == 24
    for row in body["forecasts"]:
        assert row["pred_q10"] <= row["pred_q50"] <= row["pred_q90"]


def test_forecast_unknown_plant(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    model_name = "tft-api-404"

    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    info = ProductionModelInfo(
        name=model_name,
        version="1",
        stage="Production",
        run_id="test-run",
        uri=f"models:/{model_name}/Production",
    )

    def get_model():
        return info

    def load_pyfunc():
        from energy_forecasting.model.registry import load_production_pyfunc

        return load_production_pyfunc(model_name=model_name, tracking_uri=tracking_uri)

    client = TestClient(create_app(get_model=get_model, load_pyfunc=load_pyfunc))
    response = client.post("/forecast", json={"plant_id": "DE_PV_999", "horizon": 24})
    assert response.status_code == 404


@pytest.fixture
def api_client(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    model_name = "tft-api-validation"

    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    info = ProductionModelInfo(
        name=model_name,
        version="1",
        stage="Production",
        run_id="test-run",
        uri=f"models:/{model_name}/Production",
    )

    def load_pyfunc():
        from energy_forecasting.model.registry import load_production_pyfunc

        return load_production_pyfunc(model_name=model_name, tracking_uri=tracking_uri)

    return TestClient(create_app(get_model=lambda: info, load_pyfunc=load_pyfunc))


@pytest.mark.parametrize(
    "payload",
    [
        {"plant_id": "", "horizon": 24},
        {"horizon": 24},
        {"plant_id": "DE_PV_001", "horizon": 0},
        {"plant_id": "DE_PV_001", "horizon": 48},
    ],
)
def test_forecast_rejects_bad_input(api_client: TestClient, payload: dict):
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 422


def test_health_includes_model_version(tmp_path: Path):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)
    tracking_uri = tmp_path / "mlruns"
    model_name = "tft-api-health"

    package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    info = ProductionModelInfo(
        name=model_name,
        version="1",
        stage="Production",
        run_id="test-run",
        uri=f"models:/{model_name}/Production",
    )

    client = TestClient(create_app(get_model=lambda: info))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "1"
    assert body["inference_mode"] == "batch"

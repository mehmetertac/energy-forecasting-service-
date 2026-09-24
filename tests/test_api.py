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


def _sample_forecast_parquet(path: Path, *, with_actual: bool = False) -> None:
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    data = {
        "timestamp": ts,
        "plant_id": ["DE_PV_001"] * 24,
        "horizon": list(range(1, 25)),
        "pred_q10": [3.0, 5.0, 1.0] + [float(i) for i in range(3, 24)],
        "pred_q50": [4.0] * 24,
        "pred_q90": [6.0] * 24,
    }
    if with_actual:
        data["y"] = [4.5] * 24
    pd.DataFrame(data).to_parquet(path, index=False)


def _make_client(tmp_path: Path, *, model_name: str, with_actual: bool = False):
    forecast_path = tmp_path / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path, with_actual=with_actual)
    tracking_uri = tmp_path / "mlruns"

    result = package_local_forecast(
        forecast_path=forecast_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    info = ProductionModelInfo(
        name=model_name,
        version="1",
        stage="Production",
        run_id=result.run_id,
        uri=f"models:/{model_name}/Production",
    )

    def load_pyfunc():
        from energy_forecasting.model.registry import load_production_pyfunc

        return load_production_pyfunc(model_name=model_name, tracking_uri=tracking_uri)

    def list_plants_fn():
        _m_info, model = load_pyfunc()
        inner = model.unwrap_python_model()
        return sorted(inner.forecasts["plant_id"].astype(str).unique().tolist())

    def get_metrics_fn():
        from energy_forecasting.api.schema import MetricsResponse
        from energy_forecasting.model.registry import get_production_run_metrics

        raw = get_production_run_metrics(
            run_id=info.run_id,
            model_name=model_name,
            tracking_uri=tracking_uri,
        )
        return MetricsResponse(
            model_name=info.name,
            model_version=info.version,
            run_id=info.run_id,
            pinball_q10=raw.get("mean_pinball_q10"),
            pinball_q50=raw.get("mean_pinball_q50"),
            pinball_q90=raw.get("mean_pinball_q90"),
            pi_coverage=raw.get("mean_pi_coverage"),
        )

    client = TestClient(
        create_app(
            get_model=lambda: info,
            load_pyfunc=load_pyfunc,
            list_plants=list_plants_fn,
            get_metrics=get_metrics_fn,
        )
    )
    return client, info, tracking_uri


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
    client, _info, _uri = _make_client(tmp_path, model_name="tft-api-health")
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "1"
    assert body["inference_mode"] == "batch"


def test_forecast_includes_actual_when_cached(tmp_path: Path):
    client, _info, _uri = _make_client(tmp_path, model_name="tft-api-actual", with_actual=True)
    body = client.post("/forecast", json={"plant_id": "DE_PV_001", "horizon": 24}).json()
    assert body["forecasts"][0]["actual"] == pytest.approx(4.5)


def test_plants_endpoint(tmp_path: Path):
    client, _info, _uri = _make_client(tmp_path, model_name="tft-api-plants")
    body = client.get("/plants").json()
    assert body["plants"] == ["DE_PV_001"]


def test_metrics_endpoint_nulls_without_logged_means(tmp_path: Path):
    client, _info, _uri = _make_client(tmp_path, model_name="tft-api-metrics-empty")
    body = client.get("/metrics").json()
    assert body["model_name"] == "tft-api-metrics-empty"
    assert body["pinball_q50"] is None
    assert body["pi_coverage"] is None


def test_metrics_endpoint_returns_logged_means(tmp_path: Path):
    from mlflow import MlflowClient

    client, info, tracking_uri = _make_client(tmp_path, model_name="tft-api-metrics")
    ml_client = MlflowClient(tracking_uri=tracking_uri.resolve().as_uri())
    ml_client.log_metric(info.run_id, "mean_pinball_q50", 0.21)
    ml_client.log_metric(info.run_id, "mean_pi_coverage", 0.78)

    body = client.get("/metrics").json()
    assert body["pinball_q50"] == pytest.approx(0.21)
    assert body["pi_coverage"] == pytest.approx(0.78)

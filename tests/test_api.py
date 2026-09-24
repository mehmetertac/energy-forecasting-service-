"""FastAPI batch inference contract (no torch)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from energy_forecasting.api.app import create_app
from energy_forecasting.api.schema import enforce_quantile_order
from energy_forecasting.serving.store import publish_batch


def _sample_forecast_frame(*, with_actual: bool = False) -> pd.DataFrame:
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
        data["actual"] = [4.5] * 24
    return pd.DataFrame(data)


def _publish_client(
    tmp_path: Path,
    *,
    model_name: str,
    with_actual: bool = False,
    generated_at: datetime | None = None,
) -> tuple[TestClient, Path, str]:
    db_path = tmp_path / f"{model_name}.db"
    publish_batch(
        _sample_forecast_frame(with_actual=with_actual),
        model_name=model_name,
        model_version="1",
        model_stage="Production",
        run_id=f"run-{model_name}",
        generated_at=generated_at,
        db_path=db_path,
    )
    client = TestClient(create_app(db_path=db_path))
    return client, db_path, model_name


def test_enforce_quantile_order():
    q10, q50, q90 = enforce_quantile_order(5.0, 3.0, 6.0)
    assert q10 <= q50 <= q90
    assert (q10, q50, q90) == (3.0, 5.0, 6.0)


def test_forecast_endpoint(tmp_path: Path):
    client, _db, model_name = _publish_client(tmp_path, model_name="tft-api-test")
    response = client.post("/forecast", json={"plant_id": "DE_PV_001", "horizon": 24})
    assert response.status_code == 200
    body = response.json()
    assert body["inference_mode"] == "batch"
    assert body["model_version"] == "1"
    assert body["model_stage"] == "Production"
    assert body["model_name"] == model_name
    assert len(body["forecasts"]) == 24
    for row in body["forecasts"]:
        assert row["pred_q10"] <= row["pred_q50"] <= row["pred_q90"]


def test_forecast_unknown_plant(tmp_path: Path):
    client, _db, _ = _publish_client(tmp_path, model_name="tft-api-404")
    response = client.post("/forecast", json={"plant_id": "DE_PV_999", "horizon": 24})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "unknown_plant"
    assert "request_id" in detail


def test_forecast_empty_store(tmp_path: Path):
    client = TestClient(create_app(db_path=tmp_path / "empty.db"))
    response = client.post("/forecast", json={"plant_id": "DE_PV_001", "horizon": 24})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "empty_forecasts"


def test_forecast_stale_store(tmp_path: Path):
    old = datetime.now(tz=UTC) - timedelta(hours=48)
    client, _db, _ = _publish_client(
        tmp_path,
        model_name="tft-api-stale",
        generated_at=old,
    )
    response = client.post("/forecast", json={"plant_id": "DE_PV_001", "horizon": 24})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "stale_forecast"


def test_request_id_header(tmp_path: Path):
    client, _db, _ = _publish_client(tmp_path, model_name="tft-api-reqid")
    response = client.post(
        "/forecast",
        json={"plant_id": "DE_PV_001", "horizon": 24},
        headers={"X-Request-ID": "test-req-123"},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-req-123"


@pytest.fixture
def api_client(tmp_path: Path):
    client, _db, _ = _publish_client(tmp_path, model_name="tft-api-validation")
    return client


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
    client, _db, model_name = _publish_client(tmp_path, model_name="tft-api-health")
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "1"
    assert body["inference_mode"] == "batch"
    assert body["model_name"] == model_name


def test_forecast_includes_actual_when_cached(tmp_path: Path):
    client, _db, _ = _publish_client(tmp_path, model_name="tft-api-actual", with_actual=True)
    body = client.post("/forecast", json={"plant_id": "DE_PV_001", "horizon": 24}).json()
    assert body["forecasts"][0]["actual"] == pytest.approx(4.5)


def test_plants_endpoint(tmp_path: Path):
    client, _db, _ = _publish_client(tmp_path, model_name="tft-api-plants")
    body = client.get("/plants").json()
    assert body["plants"] == ["DE_PV_001"]


def test_metrics_endpoint_nulls_without_logged_means(tmp_path: Path):
    client, _db, model_name = _publish_client(tmp_path, model_name="tft-api-metrics-empty")
    body = client.get("/metrics").json()
    assert body["model_name"] == model_name
    assert body["pinball_q50"] is None
    assert body["pi_coverage"] is None


def test_metrics_endpoint_returns_logged_means(tmp_path: Path):
    from energy_forecasting.api.schema import MetricsResponse

    client, db_path, model_name = _publish_client(tmp_path, model_name="tft-api-metrics")

    def get_metrics():
        return MetricsResponse(
            model_name=model_name,
            model_version="1",
            run_id=f"run-{model_name}",
            pinball_q50=0.21,
            pi_coverage=0.78,
        )

    client = TestClient(create_app(db_path=db_path, get_metrics=get_metrics))
    body = client.get("/metrics").json()
    assert body["pinball_q50"] == pytest.approx(0.21)
    assert body["pi_coverage"] == pytest.approx(0.78)

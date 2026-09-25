"""API schema contract: P10/P50/P90 stay in the payload."""

from __future__ import annotations

import pytest

from energy_forecasting.api.schema import ForecastResponse, QuantileForecast


def test_quantile_forecast_fields():
    row = QuantileForecast(
        timestamp="2020-06-01T12:00:00Z",
        plant_id="DE_PV_001",
        horizon=1,
        pred_q10=1.0,
        pred_q50=2.0,
        pred_q90=3.0,
    )
    dumped = row.as_dict()
    assert dumped["pred_q10"] == 1.0
    assert dumped["pred_q50"] == 2.0
    assert dumped["pred_q90"] == 3.0


def test_forecast_response_quantiles():
    payload = ForecastResponse(
        model_name="tft-solar-quantile",
        model_version="1",
        model_stage="Production",
        forecasts=[],
    )
    assert payload.quantiles == (0.1, 0.5, 0.9)


def test_create_app_health(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import pandas as pd
    from fastapi.testclient import TestClient

    from energy_forecasting.api.app import create_app
    from energy_forecasting.serving.store import publish_batch

    db_path = tmp_path / "forecasts.db"
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    publish_batch(
        pd.DataFrame(
            {
                "timestamp": ts,
                "plant_id": ["DE_PV_001"] * 24,
                "horizon": list(range(1, 25)),
                "pred_q10": [1.0] * 24,
                "pred_q50": [2.0] * 24,
                "pred_q90": [3.0] * 24,
            }
        ),
        model_name="tft-solar-quantile",
        model_version="1",
        model_stage="Production",
        run_id="schema-test",
        db_path=db_path,
    )
    client = TestClient(create_app(db_path=db_path))
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_version"] == "1"
    assert health["inference_mode"] == "batch"
    body = client.get("/schema").json()
    assert body["quantiles"] == [0.1, 0.5, 0.9]
    assert body["inference_mode"] == "batch"
    assert "model_version" in body

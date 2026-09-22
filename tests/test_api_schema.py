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


def test_create_app_health():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from energy_forecasting.api.app import create_app
    from energy_forecasting.model.registry import ProductionModelInfo

    info = ProductionModelInfo(
        name="tft-solar-quantile",
        version="1",
        stage="Production",
        run_id="schema-test",
        uri="models:/tft-solar-quantile/Production",
    )
    client = TestClient(create_app(get_model=lambda: info))
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_version"] == "1"
    assert health["inference_mode"] == "batch"
    body = client.get("/schema").json()
    assert body["quantiles"] == [0.1, 0.5, 0.9]
    assert body["inference_mode"] == "batch"
    assert "model_version" in body

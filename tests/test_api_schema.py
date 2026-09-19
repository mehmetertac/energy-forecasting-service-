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
    payload = ForecastResponse(forecasts=[])
    assert payload.quantiles == (0.1, 0.5, 0.9)


def test_create_app_health():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from energy_forecasting.api.app import create_app

    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/schema").json()
    assert body["quantiles"] == [0.1, 0.5, 0.9]

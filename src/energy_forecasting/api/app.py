"""FastAPI batch inference — serves cached P10/P50/P90 from MLflow Model Registry."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from energy_forecasting.api.schema import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    QuantileForecast,
    enforce_quantile_order,
)
from energy_forecasting.model.registry import ProductionModelInfo, get_production_model, load_production_pyfunc

logger = logging.getLogger(__name__)

GetModelFn = Callable[[], ProductionModelInfo]
LoadPyfuncFn = Callable[[], tuple[ProductionModelInfo, Any]]


def _default_get_model() -> ProductionModelInfo:
    return get_production_model()


def _default_load_pyfunc() -> tuple[ProductionModelInfo, Any]:
    return load_production_pyfunc()


def _rows_to_forecasts(rows: pd.DataFrame, plant_id: str) -> list[QuantileForecast]:
    forecasts: list[QuantileForecast] = []
    for _, row in rows.iterrows():
        q10, q50, q90 = enforce_quantile_order(
            float(row["pred_q10"]),
            float(row["pred_q50"]),
            float(row["pred_q90"]),
        )
        ts = pd.to_datetime(row["timestamp"], utc=True).isoformat().replace("+00:00", "Z")
        forecasts.append(
            QuantileForecast(
                timestamp=ts,
                plant_id=plant_id,
                horizon=int(row["horizon"]),
                pred_q10=q10,
                pred_q50=q50,
                pred_q90=q90,
            )
        )
    return forecasts


def create_app(
    *,
    get_model: GetModelFn | None = None,
    load_pyfunc: LoadPyfuncFn | None = None,
):
    """Return FastAPI app; inject ``get_model`` / ``load_pyfunc`` in tests."""
    from fastapi import FastAPI, HTTPException

    resolve_model = get_model or _default_get_model
    resolve_pyfunc = load_pyfunc or _default_load_pyfunc

    app = FastAPI(
        title="energy-forecasting-service",
        description="Batch probabilistic TFT inference (P10/P50/P90) from MLflow Model Registry.",
        version="0.2.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            info = resolve_model()
            return HealthResponse(
                status="ok",
                model_name=info.name,
                model_version=info.version,
                model_stage=info.stage,
            )
        except Exception as exc:
            logger.warning("health check failed: %s", exc)
            return HealthResponse(
                status="unhealthy",
                detail=str(exc),
            )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        try:
            info, model = resolve_pyfunc()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"production model unavailable: {exc}") from exc

        payload = {
            "plant_id": request.plant_id,
            "horizon": request.horizon,
        }
        if request.as_of is not None:
            payload["as_of"] = request.as_of

        rows = model.predict([payload])
        if isinstance(rows, pd.DataFrame) and rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"no cached forecasts for plant_id={request.plant_id!r}",
            )

        forecasts = _rows_to_forecasts(rows, request.plant_id)
        if not forecasts:
            raise HTTPException(
                status_code=404,
                detail=f"no cached forecasts for plant_id={request.plant_id!r}",
            )

        return ForecastResponse(
            model_name=info.name,
            model_version=info.version,
            model_stage=info.stage,
            forecasts=forecasts,
        )

    @app.get("/schema")
    def schema() -> dict:
        example = ForecastResponse(
            model_name="tft-solar-quantile",
            model_version="1",
            model_stage="Production",
            forecasts=[],
        ).model_dump()
        example["note"] = "POST /forecast returns pred_q10/pred_q50/pred_q90 per horizon (batch mode)."
        return example

    return app


app = create_app()

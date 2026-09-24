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
    MetricsResponse,
    PlantsResponse,
    QuantileForecast,
    enforce_quantile_order,
)
from energy_forecasting.model.registry import (
    ProductionModelInfo,
    get_production_model,
    get_production_run_metrics,
    list_cached_plants,
    load_production_pyfunc,
)

logger = logging.getLogger(__name__)

GetModelFn = Callable[[], ProductionModelInfo]
LoadPyfuncFn = Callable[[], tuple[ProductionModelInfo, Any]]
ListPlantsFn = Callable[[], list[str]]
GetMetricsFn = Callable[[], MetricsResponse]


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
        actual: float | None = None
        if "actual" in row.index and pd.notna(row["actual"]):
            actual = float(row["actual"])
        forecasts.append(
            QuantileForecast(
                timestamp=ts,
                plant_id=plant_id,
                horizon=int(row["horizon"]),
                pred_q10=q10,
                pred_q50=q50,
                pred_q90=q90,
                actual=actual,
            )
        )
    return forecasts


def _metrics_from_info(info: ProductionModelInfo) -> MetricsResponse:
    raw = get_production_run_metrics(run_id=info.run_id, model_name=info.name)
    return MetricsResponse(
        model_name=info.name,
        model_version=info.version,
        run_id=info.run_id,
        pinball_q10=raw.get("mean_pinball_q10"),
        pinball_q50=raw.get("mean_pinball_q50"),
        pinball_q90=raw.get("mean_pinball_q90"),
        pi_coverage=raw.get("mean_pi_coverage"),
    )


def create_app(
    *,
    get_model: GetModelFn | None = None,
    load_pyfunc: LoadPyfuncFn | None = None,
    list_plants: ListPlantsFn | None = None,
    get_metrics: GetMetricsFn | None = None,
):
    """Return FastAPI app; inject ``get_model`` / ``load_pyfunc`` in tests."""
    from fastapi import FastAPI, HTTPException

    resolve_model = get_model or _default_get_model
    resolve_pyfunc = load_pyfunc or _default_load_pyfunc

    def _default_list_plants() -> list[str]:
        return list_cached_plants()

    def _default_get_metrics() -> MetricsResponse:
        return _metrics_from_info(resolve_model())

    resolve_plants = list_plants or _default_list_plants
    resolve_metrics = get_metrics or _default_get_metrics

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

    @app.get("/plants", response_model=PlantsResponse)
    def plants() -> PlantsResponse:
        try:
            return PlantsResponse(plants=resolve_plants())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"production model unavailable: {exc}") from exc

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        try:
            return resolve_metrics()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"production model unavailable: {exc}") from exc

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

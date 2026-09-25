"""FastAPI batch inference — serves P10/P50/P90 from the SQLite forecast store."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from starlette.requests import Request

from energy_forecasting.api.logging_config import log_request, new_request_id
from energy_forecasting.api.schema import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetricsResponse,
    PlantsResponse,
    QuantileForecast,
    enforce_quantile_order,
)
from energy_forecasting.config import FORECAST_DB
from energy_forecasting.model.registry import get_production_run_metrics
from energy_forecasting.serving.store import (
    BatchMetadata,
    EmptyForecastError,
    StaleForecastError,
    get_batch_metadata,
    get_forecasts,
)
from energy_forecasting.serving.store import list_plants as store_list_plants

logger = logging.getLogger(__name__)

GetMetadataFn = Callable[[], BatchMetadata]
ListPlantsFn = Callable[[], list[str]]
GetMetricsFn = Callable[[], MetricsResponse]
GetForecastsFn = Callable[[str, int, str | None], tuple[BatchMetadata, pd.DataFrame]]


def _default_get_metadata() -> BatchMetadata:
    return get_batch_metadata(db_path=FORECAST_DB)


def _default_list_plants() -> list[str]:
    return store_list_plants(db_path=FORECAST_DB)


def _default_get_forecasts(plant_id: str, horizon: int, as_of: str | None) -> tuple[BatchMetadata, pd.DataFrame]:
    return get_forecasts(
        plant_id=plant_id,
        horizon=horizon,
        as_of=as_of,
        db_path=FORECAST_DB,
    )


def _default_get_metrics() -> MetricsResponse:
    meta = get_batch_metadata(db_path=FORECAST_DB, max_age_hours=float("inf"))
    try:
        raw = get_production_run_metrics(run_id=meta.run_id, model_name=meta.model_name)
    except Exception as exc:
        logger.warning("production run metrics unavailable: %s", exc)
        raw = {}
    return MetricsResponse(
        model_name=meta.model_name,
        model_version=meta.model_version,
        run_id=meta.run_id,
        pinball_q10=raw.get("mean_pinball_q10"),
        pinball_q50=raw.get("mean_pinball_q50"),
        pinball_q90=raw.get("mean_pinball_q90"),
        pi_coverage=raw.get("mean_pi_coverage"),
    )


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


def _error_detail(
    *,
    code: str,
    message: str,
    request_id: str,
    model_version: str | None = None,
) -> dict[str, str | None]:
    return {
        "code": code,
        "message": message,
        "request_id": request_id,
        "model_version": model_version,
    }


def _model_version_safe(get_metadata: GetMetadataFn) -> str | None:
    try:
        return get_metadata().model_version
    except (EmptyForecastError, StaleForecastError, FileNotFoundError):
        return None


def create_app(
    *,
    get_metadata: GetMetadataFn | None = None,
    list_plants: ListPlantsFn | None = None,
    get_metrics: GetMetricsFn | None = None,
    get_forecasts_fn: GetForecastsFn | None = None,
    db_path: Path | None = None,
):
    """Return FastAPI app; inject store helpers in tests."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    if db_path is not None:
        def _meta() -> BatchMetadata:
            return get_batch_metadata(db_path=db_path)

        def _plants() -> list[str]:
            return store_list_plants(db_path=db_path)

        def _forecasts(plant_id: str, horizon: int, as_of: str | None) -> tuple[BatchMetadata, pd.DataFrame]:
            return get_forecasts(
                plant_id=plant_id,
                horizon=horizon,
                as_of=as_of,
                db_path=db_path,
            )

        def _metrics() -> MetricsResponse:
            meta = get_batch_metadata(db_path=db_path, max_age_hours=float("inf"))
            try:
                raw = get_production_run_metrics(run_id=meta.run_id, model_name=meta.model_name)
            except Exception as exc:
                logger.warning("production run metrics unavailable: %s", exc)
                raw = {}
            return MetricsResponse(
                model_name=meta.model_name,
                model_version=meta.model_version,
                run_id=meta.run_id,
                pinball_q10=raw.get("mean_pinball_q10"),
                pinball_q50=raw.get("mean_pinball_q50"),
                pinball_q90=raw.get("mean_pinball_q90"),
                pi_coverage=raw.get("mean_pi_coverage"),
            )

        resolve_metadata = get_metadata or _meta
        resolve_plants = list_plants or _plants
        resolve_forecasts = get_forecasts_fn or _forecasts
        resolve_metrics = get_metrics or _metrics
    else:
        resolve_metadata = get_metadata or _default_get_metadata
        resolve_plants = list_plants or _default_list_plants
        resolve_forecasts = get_forecasts_fn or _default_get_forecasts
        resolve_metrics = get_metrics or _default_get_metrics

    class RequestContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request_id = request.headers.get("X-Request-ID") or new_request_id()
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            log_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                model_version=_model_version_safe(resolve_metadata),
            )
            return response

    app = FastAPI(
        title="energy-forecasting-service",
        description="Batch probabilistic TFT inference (P10/P50/P90) from SQLite forecast store.",
        version="0.3.0",
    )
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", response_model=HealthResponse)
    def health(http_request: Request) -> HealthResponse | JSONResponse:
        request_id = http_request.state.request_id
        try:
            meta = resolve_metadata()
            return HealthResponse(
                status="ok",
                model_name=meta.model_name,
                model_version=meta.model_version,
                model_stage=meta.model_stage,
            )
        except EmptyForecastError as exc:
            return JSONResponse(
                status_code=503,
                content=_error_detail(
                    code="empty_forecasts",
                    message=str(exc),
                    request_id=request_id,
                ),
            )
        except StaleForecastError as exc:
            version = _model_version_safe(lambda: get_batch_metadata(max_age_hours=float("inf")))
            return JSONResponse(
                status_code=503,
                content=_error_detail(
                    code="stale_forecast",
                    message=str(exc),
                    request_id=request_id,
                    model_version=version,
                ),
            )
        except Exception as exc:
            logger.warning("health check failed: %s", exc)
            return HealthResponse(
                status="unhealthy",
                detail=str(exc),
            )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(body: ForecastRequest, http_request: Request) -> ForecastResponse:
        request_id = http_request.state.request_id
        try:
            meta, rows = resolve_forecasts(body.plant_id, body.horizon, body.as_of)
        except EmptyForecastError as exc:
            raise HTTPException(
                status_code=503,
                detail=_error_detail(
                    code="empty_forecasts",
                    message=str(exc),
                    request_id=request_id,
                ),
            ) from exc
        except StaleForecastError as exc:
            version = _model_version_safe(lambda: get_batch_metadata(max_age_hours=float("inf")))
            raise HTTPException(
                status_code=503,
                detail=_error_detail(
                    code="stale_forecast",
                    message=str(exc),
                    request_id=request_id,
                    model_version=version,
                ),
            ) from exc

        if rows.empty:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(
                    code="unknown_plant",
                    message=f"no cached forecasts for plant_id={body.plant_id!r}",
                    request_id=request_id,
                    model_version=meta.model_version,
                ),
            )

        forecasts = _rows_to_forecasts(rows, body.plant_id)
        if not forecasts:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(
                    code="unknown_plant",
                    message=f"no cached forecasts for plant_id={body.plant_id!r}",
                    request_id=request_id,
                    model_version=meta.model_version,
                ),
            )

        return ForecastResponse(
            model_name=meta.model_name,
            model_version=meta.model_version,
            model_stage=meta.model_stage,
            forecasts=forecasts,
        )

    @app.get("/plants", response_model=PlantsResponse)
    def plants(http_request: Request) -> PlantsResponse:
        request_id = http_request.state.request_id
        try:
            resolve_metadata()
            return PlantsResponse(plants=resolve_plants())
        except EmptyForecastError as exc:
            raise HTTPException(
                status_code=503,
                detail=_error_detail(
                    code="empty_forecasts",
                    message=str(exc),
                    request_id=request_id,
                ),
            ) from exc
        except StaleForecastError as exc:
            version = _model_version_safe(lambda: get_batch_metadata(max_age_hours=float("inf")))
            raise HTTPException(
                status_code=503,
                detail=_error_detail(
                    code="stale_forecast",
                    message=str(exc),
                    request_id=request_id,
                    model_version=version,
                ),
            ) from exc

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics(http_request: Request) -> MetricsResponse:
        request_id = http_request.state.request_id
        try:
            return resolve_metrics()
        except EmptyForecastError as exc:
            raise HTTPException(
                status_code=503,
                detail=_error_detail(
                    code="empty_forecasts",
                    message=str(exc),
                    request_id=request_id,
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=_error_detail(
                    code="metrics_unavailable",
                    message=str(exc),
                    request_id=request_id,
                    model_version=_model_version_safe(resolve_metadata),
                ),
            ) from exc

    @app.get("/schema")
    def schema() -> dict[str, Any]:
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

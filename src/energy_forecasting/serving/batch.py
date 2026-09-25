"""Day-ahead batch publish pipeline — registry → SQLite."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_HORIZON, FORECAST_DB, MODEL_MAX_AGE_HOURS
from energy_forecasting.model.registry import (
    REGISTERED_MODEL_NAME,
    get_production_model,
    load_production_pyfunc,
)
from energy_forecasting.model.tft_features import KNOWN_FUTURE_REAL_COLS
from energy_forecasting.serving.store import BatchMetadata, publish_batch

logger = logging.getLogger(__name__)

REQUIRED_FORECAST_COLUMNS = (
    "timestamp",
    "plant_id",
    "horizon",
    "pred_q10",
    "pred_q50",
    "pred_q90",
)


class StaleModelError(RuntimeError):
    """Production model version is older than the allowed age."""


class MissingCovariatesError(ValueError):
    """Optional features parquet is missing required known-future columns."""


def slice_day_ahead_forecasts(
    forecasts: pd.DataFrame,
    *,
    max_horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Keep the latest origin per plant with horizons <= ``max_horizon``."""
    if forecasts.empty:
        return forecasts

    work = forecasts.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["origin"] = work["timestamp"] - pd.to_timedelta(work["horizon"], unit="h")

    parts: list[pd.DataFrame] = []
    for _, group in work.groupby("plant_id"):
        latest_origin = group["origin"].max()
        rows = group[(group["origin"] == latest_origin) & (group["horizon"] <= max_horizon)]
        parts.append(rows)

    out = pd.concat(parts, ignore_index=True)
    keep = [c for c in REQUIRED_FORECAST_COLUMNS if c in out.columns]
    if "actual" in out.columns:
        keep.append("actual")
    return out[keep].sort_values(["plant_id", "horizon", "timestamp"]).reset_index(drop=True)


def validate_forecast_frame(forecasts: pd.DataFrame) -> None:
    """Raise when the slice is empty or missing required columns."""
    if forecasts.empty:
        raise ValueError("forecast slice is empty — nothing to publish")
    missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in forecasts.columns]
    if missing:
        raise ValueError(f"forecast table missing columns: {missing}")


def validate_features_parquet(features_path: Path) -> None:
    """Ensure known-future covariate columns exist in an optional features file."""
    frame = pd.read_parquet(features_path)
    missing = [c for c in KNOWN_FUTURE_REAL_COLS if c not in frame.columns]
    if missing:
        raise MissingCovariatesError(f"features parquet missing known-future columns: {missing}")


def assert_model_fresh(
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    model_version: str,
    tracking_uri: str | Path | None = None,
    max_age_hours: float | None = None,
) -> None:
    """Refuse to publish when the Production version is older than ``max_age_hours``."""
    from energy_forecasting.model.registry import _client

    client = _client(tracking_uri)
    version_info = client.get_model_version(model_name, model_version)
    created_ms = version_info.creation_timestamp
    created_at = datetime.fromtimestamp(created_ms / 1000.0, tz=UTC)
    limit = MODEL_MAX_AGE_HOURS if max_age_hours is None else max_age_hours
    age = datetime.now(tz=UTC) - created_at
    if age > timedelta(hours=limit):
        raise StaleModelError(
            f"production model {model_name} v{model_version} is stale "
            f"(age={age}, max={timedelta(hours=limit)})"
        )


def _forecasts_from_pyfunc(model) -> pd.DataFrame:
    inner = model.unwrap_python_model()
    return inner.forecasts.copy()


def publish_day_ahead_forecasts(
    *,
    db_path: Path | str | None = None,
    features_path: Path | None = None,
    max_horizon: int = DEFAULT_HORIZON,
    model_max_age_hours: float | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    tracking_uri: str | Path | None = None,
    seed_if_missing: bool = False,
) -> BatchMetadata:
    """Load Production pyfunc, slice day-ahead rows, validate, and publish to SQLite."""
    if features_path is not None:
        validate_features_parquet(features_path)

    try:
        info = get_production_model(model_name=model_name, tracking_uri=tracking_uri)
    except RuntimeError as exc:
        if not seed_if_missing:
            raise
        logger.warning("no production model — seeding synthetic forecasts: %s", exc)
        from energy_forecasting.serving.seed import seed_production_model

        seed_production_model(tracking_uri=tracking_uri)
        info = get_production_model(model_name=model_name, tracking_uri=tracking_uri)

    assert_model_fresh(
        model_name=info.name,
        model_version=info.version,
        tracking_uri=tracking_uri,
        max_age_hours=model_max_age_hours,
    )

    _info, model = load_production_pyfunc(
        model_name=info.name,
        stage=info.stage,
        tracking_uri=tracking_uri,
    )
    raw = _forecasts_from_pyfunc(model)
    sliced = slice_day_ahead_forecasts(raw, max_horizon=max_horizon)
    validate_forecast_frame(sliced)

    meta = publish_batch(
        sliced,
        model_name=info.name,
        model_version=info.version,
        model_stage=info.stage,
        run_id=info.run_id,
        db_path=db_path or FORECAST_DB,
    )
    logger.info(
        "published day-ahead forecasts",
        extra={
            "model_name": meta.model_name,
            "model_version": meta.model_version,
            "plants": int(sliced["plant_id"].nunique()),
            "rows": len(sliced),
            "db_path": str(db_path or FORECAST_DB),
        },
    )
    return meta

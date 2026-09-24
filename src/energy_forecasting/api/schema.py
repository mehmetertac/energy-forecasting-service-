"""P10/P50/P90 forecast payload for batch inference API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KnownFutureCovariates(BaseModel):
    """Optional known-future covariates for a future on-demand inference path."""

    model_config = ConfigDict(extra="allow")

    timestamp: str


class ForecastRequest(BaseModel):
    """Request for day-ahead quantile forecasts."""

    plant_id: str = Field(..., description="Plant / series identifier (e.g. DE_PV_001).")
    horizon: int = Field(default=24, ge=1, le=24, description="Forecast horizon in hours (1–24).")
    as_of: str | None = Field(
        default=None,
        description="Optional UTC anchor; batch mode uses the latest cached origin when omitted.",
    )
    known_future: list[KnownFutureCovariates] | None = Field(
        default=None,
        description="Optional known-future covariates (ignored in batch mode; reserved for on-demand).",
    )

    @field_validator("plant_id")
    @classmethod
    def strip_plant_id(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("plant_id must not be empty")
        return text


class QuantileForecast(BaseModel):
    """Single plant-horizon probabilistic forecast."""

    timestamp: str
    plant_id: str
    horizon: int = Field(ge=1)
    pred_q10: float
    pred_q50: float
    pred_q90: float
    actual: float | None = Field(default=None, description="Observed MW when available in the cached table.")

    def as_dict(self) -> dict[str, str | int | float | None]:
        return self.model_dump()


class ForecastResponse(BaseModel):
    """API envelope for a batch of quantile forecasts."""

    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    inference_mode: str = "batch"
    model_name: str
    model_version: str
    model_stage: str
    forecasts: list[QuantileForecast]


class HealthResponse(BaseModel):
    """Service health with resolved production model identity."""

    status: str
    inference_mode: str = "batch"
    model_name: str | None = None
    model_version: str | None = None
    model_stage: str | None = None
    detail: str | None = None


class PlantsResponse(BaseModel):
    """Plant ids available in the production forecast cache."""

    plants: list[str]


class MetricsResponse(BaseModel):
    """Recent validation metrics from the production model run."""

    model_name: str
    model_version: str
    run_id: str
    pinball_q10: float | None = None
    pinball_q50: float | None = None
    pinball_q90: float | None = None
    pi_coverage: float | None = None


def enforce_quantile_order(q10: float, q50: float, q90: float) -> tuple[float, float, float]:
    """Clip quantiles so P10 <= P50 <= P90 (isotonic on three points)."""
    ordered = sorted([q10, q50, q90])
    return ordered[0], ordered[1], ordered[2]

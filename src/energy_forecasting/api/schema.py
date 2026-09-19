"""P10/P50/P90 forecast payload used by the future inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuantileForecast(BaseModel):
    """Single plant-horizon probabilistic forecast."""

    timestamp: str
    plant_id: str
    horizon: int = Field(ge=1)
    pred_q10: float
    pred_q50: float
    pred_q90: float

    def as_dict(self) -> dict[str, str | int | float]:
        return self.model_dump()


class ForecastResponse(BaseModel):
    """API envelope for a batch of quantile forecasts."""

    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    forecasts: list[QuantileForecast]

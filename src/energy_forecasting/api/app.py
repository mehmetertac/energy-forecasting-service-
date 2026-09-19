"""FastAPI stub — inference wiring lands in a later Week 9 task."""

from __future__ import annotations

from energy_forecasting.api.schema import ForecastResponse


def create_app():
    """Return a minimal FastAPI app with health + schema placeholders."""
    from fastapi import FastAPI

    app = FastAPI(
        title="energy-forecasting-service",
        description="Probabilistic TFT inference (P10/P50/P90). Serving not wired yet.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/schema")
    def schema() -> dict:
        example = ForecastResponse(forecasts=[]).model_dump()
        example["note"] = "POST /predict will return pred_q10/pred_q50/pred_q90 per horizon."
        return example

    return app

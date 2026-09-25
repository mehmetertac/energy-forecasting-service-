"""Seed a synthetic Production model for Docker / compose smoke tests."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from energy_forecasting.model.registry import REGISTERED_MODEL_NAME, _default_tracking_uri, package_local_forecast


def sample_forecast_parquet(path: Path) -> None:
    """Write a minimal 24h quantile forecast table for one plant."""
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "plant_id": ["DE_PV_001"] * 24,
            "horizon": list(range(1, 25)),
            "pred_q10": [float(i) for i in range(24)],
            "pred_q50": [float(i + 1) for i in range(24)],
            "pred_q90": [float(i + 2) for i in range(24)],
            "y": [float(i + 1.5) for i in range(24)],
        }
    )
    df.to_parquet(path, index=False)


def _chmod_tree(root: Path) -> None:
    for dirpath, _dirs, files in os.walk(root):
        os.chmod(dirpath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for name in files:
            path = Path(dirpath) / name
            mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
            os.chmod(path, mode)


def seed_production_model(
    *,
    store: Path | None = None,
    tracking_uri: str | Path | None = None,
) -> str:
    """Register tft-solar-quantile Production and return model URI."""
    uri = _default_tracking_uri(tracking_uri or store)
    with tempfile.TemporaryDirectory() as tmp:
        forecast_path = Path(tmp) / "forecasts.parquet"
        sample_forecast_parquet(forecast_path)

        result = package_local_forecast(
            forecast_path=forecast_path,
            model_name=REGISTERED_MODEL_NAME,
            tracking_uri=uri,
            run_name="docker-seed",
        )

    client = MlflowClient(tracking_uri=uri)
    for key, value in {
        "mean_pinball_q10": 0.42,
        "mean_pinball_q50": 0.21,
        "mean_pinball_q90": 0.38,
        "mean_pi_coverage": 0.78,
    }.items():
        client.log_metric(result.run_id, key, value)

    if store is not None:
        _chmod_tree(store)

    return result.uri

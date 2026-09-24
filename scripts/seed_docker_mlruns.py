"""Seed a local MLflow store with tft-solar-quantile Production for Docker smoke tests."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

import pandas as pd

from energy_forecasting.model.registry import REGISTERED_MODEL_NAME, package_local_forecast


def _sample_forecast_parquet(path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("docker/mlruns-seed"),
        help="MLflow file store directory to create (default: docker/mlruns-seed)",
    )
    args = parser.parse_args()

    args.store.mkdir(parents=True, exist_ok=True)
    forecast_path = args.store / "forecasts.parquet"
    _sample_forecast_parquet(forecast_path)

    result = package_local_forecast(
        forecast_path=forecast_path,
        model_name=REGISTERED_MODEL_NAME,
        tracking_uri=args.store,
        run_name="docker-seed",
    )

    from mlflow import MlflowClient

    tracking_uri = args.store.resolve().as_uri()
    client = MlflowClient(tracking_uri=tracking_uri)
    for key, value in {
        "mean_pinball_q10": 0.42,
        "mean_pinball_q50": 0.21,
        "mean_pinball_q90": 0.38,
        "mean_pi_coverage": 0.78,
    }.items():
        client.log_metric(result.run_id, key, value)
    for root, _dirs, files in os.walk(args.store):
        os.chmod(root, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for name in files:
            path = Path(root) / name
            mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
            os.chmod(path, mode)

    print(f"seeded {result.uri} at {args.store.resolve()}")


if __name__ == "__main__":
    main()

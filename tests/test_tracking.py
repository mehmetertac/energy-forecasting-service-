"""MLflow tracker writes params/metrics to a temp URI (no TFT fit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.model.plots import plot_quantile_bands
from energy_forecasting.model.tracking import ExperimentTracker


def test_tracker_logs_params_and_fold_metrics(tmp_path: Path):
    tracker = ExperimentTracker(experiment_name="unit-test-tft", tracking_uri=tmp_path / "mlruns")
    tracker.start_run(
        "unit-run",
        {"horizon": 24, "hidden_size": 16, "quantiles": [0.1, 0.5, 0.9]},
    )
    tracker.log_fold_metrics(
        1,
        {
            "pinball_q10": 0.2,
            "pinball_q50": 0.1,
            "pinball_q90": 0.3,
            "crps": 0.15,
            "pi_coverage": 0.8,
        },
    )
    tracker.log_mean_metrics(
        pd.DataFrame(
            [
                {"pinball_q50": 0.1, "crps": 0.15, "pi_coverage": 0.8},
                {"pinball_q50": 0.2, "crps": 0.25, "pi_coverage": 0.7},
            ]
        )
    )
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("fold,pinball_q50\n1,0.1\n", encoding="utf-8")
    tracker.log_artifact(artifact)
    tracker.end_run()

    runs_root = tmp_path / "mlruns"
    assert any(runs_root.rglob("params/horizon"))
    assert any(runs_root.rglob("metrics/fold_1_pinball_q50"))


def test_plot_quantile_bands(tmp_path: Path):
    ts = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    oof = pd.DataFrame(
        {
            "timestamp": ts,
            "plant_id": "DE_PV_001",
            "horizon": 1,
            "power_mw": 5.0,
            "pred_q10": 3.0,
            "pred_q50": 5.0,
            "pred_q90": 7.0,
        }
    )
    path = plot_quantile_bands(oof, tmp_path / "bands.png")
    assert path.exists()
    assert path.stat().st_size > 0

"""Unit tests for probabilistic metrics."""

from __future__ import annotations

import numpy as np
import pytest

from energy_forecasting.model.metrics import (
    crps_from_quantiles,
    evaluate_quantile_forecast,
    filter_horizon,
    pinball_loss,
    pi_coverage,
)
import pandas as pd


def test_pinball_perfect_median():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)


def test_pinball_rejects_bad_quantile():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 0.0)


def test_pi_coverage_inside_band():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([4.0, 4.0, 4.0])
    assert pi_coverage(y, lo, hi) == pytest.approx(1.0)


def test_pi_coverage_misses():
    y = np.array([10.0, 10.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    assert pi_coverage(y, lo, hi) == pytest.approx(0.0)


def test_evaluate_quantile_forecast_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    preds = {
        0.1: y - 0.5,
        0.5: y,
        0.9: y + 0.5,
    }
    metrics = evaluate_quantile_forecast(y, preds)
    assert metrics["pinball_q50"] == pytest.approx(0.0)
    assert metrics["pi_coverage"] == pytest.approx(1.0)
    assert metrics["crps"] >= 0.0
    assert set(metrics) >= {"pinball_q10", "pinball_q50", "pinball_q90", "pi_coverage", "crps"}


def test_crps_from_quantiles_nonnegative():
    y = np.array([0.0, 1.0, 2.0])
    preds = {0.1: y - 1, 0.5: y, 0.9: y + 1}
    assert crps_from_quantiles(y, [0.1, 0.5, 0.9], preds) >= 0.0


def test_filter_horizon():
    df = pd.DataFrame({"horizon": [1, 2, 1], "pred_q50": [1.0, 2.0, 3.0]})
    out = filter_horizon(df, horizon=1)
    assert len(out) == 2
    assert (out["horizon"] == 1).all()

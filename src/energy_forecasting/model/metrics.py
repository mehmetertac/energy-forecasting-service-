"""Probabilistic and point forecast evaluation metrics."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecasting.config import ARTIFACTS_DIR, DEFAULT_QUANTILES

logger = logging.getLogger(__name__)

PI_COVERAGE_TARGET = 0.80

METRICS_TABLE_COLUMNS: tuple[str, ...] = (
    "fold",
    "n_train",
    "n_test",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "pi_coverage",
    "crps",
    "mae",
    "rmse",
    "mape",
)


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    quantile: float,
) -> float:
    """Mean pinball (quantile) loss at ``quantile`` in (0, 1)."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def pi_coverage(
    y_true: np.ndarray | pd.Series,
    p_lo: np.ndarray | pd.Series,
    p_hi: np.ndarray | pd.Series,
) -> float:
    """Fraction of observations inside the prediction interval [P10, P90]."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(p_lo, dtype=float)
    hi = np.asarray(p_hi, dtype=float)
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def crps(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """CRPS for a deterministic (point) forecast."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def crps_from_quantiles(
    y_true: np.ndarray | pd.Series,
    quantiles: Sequence[float],
    predictions: dict[float, np.ndarray | pd.Series],
) -> float:
    """Approximate CRPS by integrating pinball loss over quantile levels."""
    qs = sorted(set(float(q) for q in quantiles))
    if len(qs) < 2:
        q_med = qs[0]
        return crps(y_true, predictions[q_med])

    losses = []
    widths = []
    for q_lo, q_hi in zip(qs[:-1], qs[1:]):
        width = q_hi - q_lo
        q_mid = 0.5 * (q_lo + q_hi)
        yq = np.asarray(predictions[q_mid if q_mid in predictions else q_lo], dtype=float)
        losses.append(pinball_loss(y_true, yq, q_mid))
        widths.append(width)
    return float(np.dot(losses, widths) / np.sum(widths))


def evaluate_quantile_forecast(
    y_true: np.ndarray | pd.Series,
    predictions: dict[float, np.ndarray | pd.Series],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, float]:
    """Compute pinball, PI coverage, CRPS, and P50 point errors."""
    yt = np.asarray(y_true, dtype=float)
    q_lo, q_med, q_hi = quantiles
    y_med = np.asarray(predictions[q_med], dtype=float)

    metrics: dict[str, float] = {
        "n_test": float(len(yt)),
        "mae": mae(yt, y_med),
        "rmse": rmse(yt, y_med),
        "mape": mape(yt, y_med),
        "crps": crps_from_quantiles(yt, quantiles, predictions),
    }
    for q in quantiles:
        metrics[f"pinball_q{int(q * 100):02d}"] = pinball_loss(yt, predictions[q], q)
    if q_lo in predictions and q_hi in predictions:
        metrics["pi_coverage"] = pi_coverage(yt, predictions[q_lo], predictions[q_hi])
    return metrics


def format_metrics_table(
    fold_metrics: pd.DataFrame,
    *,
    include_mean: bool = True,
) -> pd.DataFrame:
    table = fold_metrics.copy()
    for col in METRICS_TABLE_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    table = table[list(METRICS_TABLE_COLUMNS)]
    if include_mean and not table.empty:
        numeric = table.select_dtypes(include="number")
        mean_row = numeric.mean(numeric_only=True).to_dict()
        mean_row["fold"] = 0
        if "n_train" in mean_row:
            mean_row["n_train"] = np.nan
        table = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)
    return table


def save_metrics_table(
    fold_metrics: pd.DataFrame,
    path: Path | str | None = None,
    *,
    include_mean: bool = True,
) -> Path:
    out_path = Path(path or ARTIFACTS_DIR / "metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = format_metrics_table(fold_metrics, include_mean=include_mean)
    table.to_csv(out_path, index=False, float_format="%.6f")
    logger.info("saved metrics table to %s (%d rows)", out_path, len(table))
    return out_path


def filter_horizon(oof: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Keep OOF rows for a single forecast horizon."""
    if "horizon" not in oof.columns:
        return oof
    return oof.loc[oof["horizon"] == horizon].copy()

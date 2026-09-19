#!/usr/bin/env python
"""Train global TFT with rolling-origin CV and MLflow tracking."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.add_dll_directory(r"C:\Windows\System32")

import torch  # preload before lightning (via TFTForecaster import)

import pandas as pd

from energy_forecasting.config import (
    CV_N_SPLITS,
    DEFAULT_HORIZON,
    DEFAULT_QUANTILES,
    METRICS_TFT_PLANT_CSV,
    OOF_TFT_PARQUET,
    TARGET_COL,
    TFT_BATCH_SIZE,
    TFT_DROPOUT,
    TFT_ENCODER_LENGTH,
    TFT_HIDDEN_SIZE,
    TFT_LEARNING_RATE,
    TFT_MAX_EPOCHS,
    TFT_MIN_ENCODER_LENGTH,
    TIMESTAMP_COL,
)
from energy_forecasting.data import load_plants_hourly, load_plants_metadata
from energy_forecasting.model.cv import RollingOriginSplit
from energy_forecasting.model.metrics import evaluate_quantile_forecast, filter_horizon, save_metrics_table
from energy_forecasting.model.plots import plot_quantile_bands
from energy_forecasting.model.tft import TFTForecaster
from energy_forecasting.model.tft_features import default_tft_feature_columns, prepare_tft_frame
from energy_forecasting.model.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


def _evaluate_plant_fold(oof_h1: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    te = test_df.dropna(subset=[TARGET_COL])
    if te.empty or oof_h1.empty:
        return {"n_test": 0.0}
    pred_cols = [f"pred_q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]
    merged = te[[TIMESTAMP_COL, TARGET_COL, "plant_id"]].merge(
        oof_h1[[TIMESTAMP_COL, "plant_id", *pred_cols]],
        on=[TIMESTAMP_COL, "plant_id"],
        how="inner",
    )
    if merged.empty:
        return {"n_test": 0.0}
    preds = {
        q: merged[f"pred_q{int(q * 100):02d}"].to_numpy(dtype=float)
        for q in DEFAULT_QUANTILES
    }
    return evaluate_quantile_forecast(merged[TARGET_COL], preds, quantiles=DEFAULT_QUANTILES)


def run_tft_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_splits: int = CV_N_SPLITS,
    gap: int = DEFAULT_HORIZON,
    forecaster: TFTForecaster | None = None,
    tracker: ExperimentTracker | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rolling-origin CV; return plant fold metrics and multi-horizon OOF frame."""
    model = forecaster or TFTForecaster()
    splitter = RollingOriginSplit(n_splits=n_splits, gap=gap, time_col=TIMESTAMP_COL)

    fold_rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    for train_df, test_df, fold in splitter.split_frame(df, time_col=TIMESTAMP_COL):
        cols = list(feature_cols)
        tr = train_df.dropna(subset=[*cols, TARGET_COL])
        te = test_df.dropna(subset=[*cols, TARGET_COL])
        if tr.empty or te.empty:
            continue

        oof_fold = model.predict_fold_oof(tr, te, fold=fold.fold)
        oof_parts.append(oof_fold)
        h1 = filter_horizon(oof_fold, horizon=1)
        metrics = _evaluate_plant_fold(h1, te)
        row = {
            "fold": fold.fold,
            "n_train": fold.n_train,
            "n_test": fold.n_test,
            **{k: float(v) for k, v in metrics.items()},
        }
        fold_rows.append(row)
        if tracker is not None:
            tracker.log_fold_metrics(fold.fold, metrics)
        logger.info(
            "fold %d complete — n_test=%s mae=%.4f pinball_q50=%.4f pi_coverage=%.3f",
            fold.fold,
            metrics.get("n_test"),
            metrics.get("mae", float("nan")),
            metrics.get("pinball_q50", float("nan")),
            metrics.get("pi_coverage", float("nan")),
        )

    if not fold_rows:
        raise ValueError("no fold metrics produced — check timestamp coverage and n_splits")

    fold_metrics = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return fold_metrics, oof


def save_oof_parquet(oof: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        "ds",
        "unique_id",
        "plant_id",
        "horizon",
        "fold",
        "y",
        TARGET_COL,
        "pred_q10",
        "pred_q50",
        "pred_q90",
        "model",
    ]
    out = oof.copy()
    if "y" not in out.columns and TARGET_COL in out.columns:
        out["y"] = out[TARGET_COL]
    if "ds" not in out.columns and TIMESTAMP_COL in out.columns:
        out["ds"] = out[TIMESTAMP_COL]
    cols = [c for c in keep if c in out.columns]
    out[cols].to_parquet(path, index=False)
    logger.info("saved TFT forecasts to %s (%d rows)", path, len(out))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TFT with rolling-origin CV and MLflow.")
    parser.add_argument("--n-splits", type=int, default=CV_N_SPLITS)
    parser.add_argument("--gap", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-epochs", type=int, default=TFT_MAX_EPOCHS)
    parser.add_argument("--hidden-size", type=int, default=TFT_HIDDEN_SIZE)
    parser.add_argument("--learning-rate", type=float, default=TFT_LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=TFT_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=TFT_BATCH_SIZE)
    parser.add_argument("--encoder-length", type=int, default=TFT_ENCODER_LENGTH)
    parser.add_argument("--min-encoder-length", type=int, default=TFT_MIN_ENCODER_LENGTH)
    parser.add_argument("--max-plants", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_TFT_PLANT_CSV)
    parser.add_argument("--output", type=Path, default=OOF_TFT_PARQUET)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Mirror this run to Weights & Biases.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow (debug only).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    plants_hourly = load_plants_hourly()
    plants_metadata = load_plants_metadata()
    if args.max_plants is not None:
        keep = sorted(plants_metadata["plant_id"].unique())[: args.max_plants]
        plants_metadata = plants_metadata[plants_metadata["plant_id"].isin(keep)]
        plants_hourly = plants_hourly[plants_hourly["plant_id"].isin(keep)]

    df = prepare_tft_frame(plants_hourly, plants_metadata)
    feature_cols = default_tft_feature_columns(df)
    logger.info(
        "modeling frame: %d rows, %d plants, %d features",
        len(df),
        df["plant_id"].nunique(),
        len(feature_cols),
    )

    params = {
        "horizon": DEFAULT_HORIZON,
        "hidden_size": args.hidden_size,
        "quantiles": DEFAULT_QUANTILES,
        "encoder_length": args.encoder_length,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "n_splits": args.n_splits,
        "max_plants": args.max_plants or df["plant_id"].nunique(),
    }
    run_name = args.run_name or f"tft-h{args.hidden_size}-e{args.max_epochs}"
    tracker = None
    if not args.no_mlflow:
        tracker = ExperimentTracker(enable_wandb=args.wandb)
        tracker.start_run(run_name, params)

    try:
        forecaster = TFTForecaster(
            max_epochs=args.max_epochs,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            batch_size=args.batch_size,
            max_encoder_length=args.encoder_length,
            min_encoder_length=args.min_encoder_length,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        fold_metrics, oof = run_tft_cv(
            df,
            feature_cols,
            n_splits=args.n_splits,
            gap=args.gap,
            forecaster=forecaster,
            tracker=tracker,
        )
        save_metrics_table(fold_metrics, args.metrics_output)
        save_oof_parquet(oof, args.output)
        plot_path = plot_quantile_bands(oof)

        if tracker is not None:
            tracker.log_mean_metrics(fold_metrics)
            tracker.log_artifact(args.metrics_output)
            tracker.log_artifact(args.output)
            tracker.log_artifact(plot_path)
            ckpt = forecaster.latest_checkpoint()
            if ckpt is not None:
                tracker.log_artifact(ckpt)
    finally:
        if tracker is not None:
            tracker.end_run()

    print("TFT training complete.")
    print(f"  Plant metrics: {args.metrics_output}")
    print(f"  OOF forecasts: {args.output}")
    print("  MLflow UI:     mlflow ui --backend-store-uri ./mlruns")
    return 0


if __name__ == "__main__":
    sys.exit(main())

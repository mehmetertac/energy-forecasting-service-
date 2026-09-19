"""Temporal Fusion Transformer forecaster (pytorch-forecasting)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    os.add_dll_directory(r"C:\Windows\System32")

import torch
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from energy_forecasting.config import (
    DEFAULT_QUANTILES,
    TARGET_COL,
    TFT_ATTENTION_HEAD_SIZE,
    TFT_BATCH_SIZE,
    TFT_CHECKPOINT_DIR,
    TFT_DROPOUT,
    TFT_EARLY_STOPPING_PATIENCE,
    TFT_ENCODER_LENGTH,
    TFT_HIDDEN_SIZE,
    TFT_LEARNING_RATE,
    TFT_MAX_EPOCHS,
    TFT_MIN_ENCODER_LENGTH,
    TFT_PREDICTION_LENGTH,
    TIMESTAMP_COL,
)
from energy_forecasting.model.tft_features import TFTFeatureRoles, resolve_feature_roles

logger = logging.getLogger(__name__)

GROUP_COL = "plant_id"
TIME_IDX_COL = "time_idx"


class TFTForecaster:
    """Global multi-series TFT with quantile outputs for plant-level solar forecasting."""

    def __init__(
        self,
        *,
        quantiles: list[float] | None = None,
        max_encoder_length: int = TFT_ENCODER_LENGTH,
        min_encoder_length: int = TFT_MIN_ENCODER_LENGTH,
        max_prediction_length: int = TFT_PREDICTION_LENGTH,
        hidden_size: int = TFT_HIDDEN_SIZE,
        attention_head_size: int = TFT_ATTENTION_HEAD_SIZE,
        dropout: float = TFT_DROPOUT,
        learning_rate: float = TFT_LEARNING_RATE,
        batch_size: int = TFT_BATCH_SIZE,
        max_epochs: int = TFT_MAX_EPOCHS,
        early_stopping_patience: int = TFT_EARLY_STOPPING_PATIENCE,
        checkpoint_dir: Path | str | None = TFT_CHECKPOINT_DIR,
        accelerator: str = "auto",
        limit_train_batches: int | None = None,
        limit_val_batches: int | None = None,
    ) -> None:
        self.quantiles = list(quantiles or DEFAULT_QUANTILES)
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.accelerator = accelerator
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        self.model: TemporalFusionTransformer | None = None
        self.training_dataset: TimeSeriesDataSet | None = None
        self.roles: TFTFeatureRoles | None = None
        self._time_idx_to_ts: dict[int, pd.Timestamp] = {}

    def _build_dataset(
        self,
        df: pd.DataFrame,
        *,
        max_time_idx: int | None = None,
        predict: bool = False,
        min_prediction_idx: int | None = None,
    ) -> TimeSeriesDataSet:
        if self.roles is None:
            self.roles = resolve_feature_roles(df)

        roles = self.roles
        work = df
        if max_time_idx is not None:
            work = df[df[TIME_IDX_COL] <= max_time_idx].copy()

        kwargs: dict[str, Any] = dict(
            time_idx=TIME_IDX_COL,
            target=TARGET_COL,
            group_ids=[GROUP_COL],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            min_encoder_length=self.min_encoder_length,
            static_categoricals=list(roles.static_categoricals),
            static_reals=list(roles.static_reals),
            time_varying_known_reals=list(roles.known_future_reals),
            time_varying_unknown_reals=list(roles.observed_unknown_reals),
            target_normalizer=GroupNormalizer(groups=[GROUP_COL]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        if min_prediction_idx is not None:
            kwargs["min_prediction_idx"] = min_prediction_idx

        if predict and self.training_dataset is not None:
            return TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                df,
                predict=True,
                stop_randomization=True,
                min_prediction_idx=min_prediction_idx,
            )

        return TimeSeriesDataSet(work, **kwargs)

    def fit(self, train_df: pd.DataFrame) -> TemporalFusionTransformer:
        """Fit TFT on training rows; keeps validation tail inside train period."""
        self.roles = resolve_feature_roles(train_df)
        self._time_idx_to_ts = (
            train_df[[TIME_IDX_COL, TIMESTAMP_COL]]
            .drop_duplicates(TIME_IDX_COL)
            .set_index(TIME_IDX_COL)[TIMESTAMP_COL]
            .to_dict()
        )

        max_idx = int(train_df[TIME_IDX_COL].max())
        val_cutoff = max_idx - self.max_prediction_length
        if val_cutoff <= int(train_df[TIME_IDX_COL].min()) + self.min_encoder_length:
            val_cutoff = max_idx

        self.training_dataset = self._build_dataset(train_df, max_time_idx=val_cutoff)
        validation = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            train_df,
            min_prediction_idx=val_cutoff + 1,
            predict=True,
            stop_randomization=True,
        )

        train_loader = self.training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=self.batch_size * 2, num_workers=0
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            loss=QuantileLoss(self.quantiles),
            log_interval=-1,
            reduce_on_plateau_patience=4,
        )

        callbacks: list[pl.Callback] = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                mode="min",
            ),
        ]
        ckpt_path: str | None = None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    filename="tft-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                )
            )
            ckpt_path = str(self.checkpoint_dir)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            enable_model_summary=False,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            default_root_dir=ckpt_path,
            logger=False,
            enable_checkpointing=self.checkpoint_dir is not None,
            limit_train_batches=self.limit_train_batches or 1.0,
            limit_val_batches=self.limit_val_batches or 1.0,
        )
        trainer.fit(self.model, train_loader, val_loader)
        return self.model

    def load_checkpoint(self, checkpoint_path: Path | str) -> TemporalFusionTransformer:
        """Restore a fitted TFT from a Lightning checkpoint."""
        if self.training_dataset is None:
            raise RuntimeError(
                "training_dataset missing — call fit() or prepare_training_dataset() first"
            )

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
        return self.model

    def prepare_training_dataset(self, train_df: pd.DataFrame) -> TimeSeriesDataSet:
        """Build ``training_dataset`` without running the Lightning trainer."""
        self.roles = resolve_feature_roles(train_df)
        self._time_idx_to_ts = (
            train_df[[TIME_IDX_COL, TIMESTAMP_COL]]
            .drop_duplicates(TIME_IDX_COL)
            .set_index(TIME_IDX_COL)[TIMESTAMP_COL]
            .to_dict()
        )
        max_idx = int(train_df[TIME_IDX_COL].max())
        val_cutoff = max_idx - self.max_prediction_length
        if val_cutoff <= int(train_df[TIME_IDX_COL].min()) + self.min_encoder_length:
            val_cutoff = max_idx
        self.training_dataset = self._build_dataset(train_df, max_time_idx=val_cutoff)
        return self.training_dataset

    def _predict_tensor(
        self,
        dataset: TimeSeriesDataSet,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if self.model is None:
            raise RuntimeError("model is not fitted — call fit() first")

        loader = dataset.to_dataloader(train=False, batch_size=self.batch_size * 2, num_workers=0)
        with torch.no_grad():
            result = self.model.predict(loader, mode="quantiles", return_index=True)

        if hasattr(result, "output"):
            predictions = result.output
            index = result.index
        elif isinstance(result, tuple) and len(result) == 2:
            predictions, index = result
        else:
            raise RuntimeError("unexpected predict() return format from TFT")

        if isinstance(predictions, torch.Tensor):
            pred_arr = predictions.detach().cpu().numpy()
        else:
            pred_arr = np.asarray(predictions)

        if index is None:
            raise RuntimeError("predict() did not return an index frame")

        if not isinstance(index, pd.DataFrame):
            index = pd.DataFrame(index)

        return pred_arr, index.reset_index(drop=True)

    def _flatten_predictions(
        self,
        pred_arr: np.ndarray,
        index: pd.DataFrame,
        *,
        context_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Expand (sample, horizon, quantile) tensor to long-format rows."""
        ts_map = (
            context_df[[TIME_IDX_COL, TIMESTAMP_COL]]
            .drop_duplicates(TIME_IDX_COL)
            .set_index(TIME_IDX_COL)[TIMESTAMP_COL]
            .to_dict()
        )
        uid_map = context_df.groupby(GROUP_COL)["unique_id"].first().to_dict()

        q_cols = [f"pred_q{int(q * 100):02d}" for q in self.quantiles]
        rows: list[dict[str, Any]] = []

        for i in range(len(index)):
            origin_idx = int(index.loc[i, TIME_IDX_COL])
            plant_id = str(index.loc[i, GROUP_COL])
            for h in range(pred_arr.shape[1]):
                target_idx = origin_idx + h + 1
                if target_idx not in ts_map:
                    continue
                row: dict[str, Any] = {
                    "ds": ts_map[target_idx],
                    TIMESTAMP_COL: ts_map[target_idx],
                    GROUP_COL: plant_id,
                    "unique_id": uid_map.get(plant_id, plant_id),
                    "horizon": h + 1,
                    TARGET_COL: np.nan,
                }
                for q_i, q in enumerate(self.quantiles):
                    row[q_cols[q_i]] = float(pred_arr[i, h, q_i])
                rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=[TIMESTAMP_COL, "ds", GROUP_COL, "unique_id", "horizon", TARGET_COL, *q_cols]
            )

        out = pd.DataFrame(rows)
        actuals = context_df[[GROUP_COL, TIMESTAMP_COL, TARGET_COL]].drop_duplicates()
        out = out.merge(actuals, on=[GROUP_COL, TIMESTAMP_COL], how="left", suffixes=("", "_actual"))
        if f"{TARGET_COL}_actual" in out.columns:
            out[TARGET_COL] = out[f"{TARGET_COL}_actual"].combine_first(out[TARGET_COL])
            out = out.drop(columns=[f"{TARGET_COL}_actual"])
        out["ds"] = pd.to_datetime(out["ds"], utc=True)
        out[TIMESTAMP_COL] = pd.to_datetime(out[TIMESTAMP_COL], utc=True)
        return out

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        *,
        max_train_time_idx: int,
        min_prediction_idx: int | None = None,
    ) -> pd.DataFrame:
        """Predict all horizons for origins after ``max_train_time_idx``."""
        if self.training_dataset is None:
            raise RuntimeError("training dataset missing — call fit() first")

        if min_prediction_idx is None:
            min_prediction_idx = max_train_time_idx + 1

        pred_ds = self._build_dataset(
            df,
            predict=True,
            min_prediction_idx=min_prediction_idx,
        )
        pred_arr, index = self._predict_tensor(pred_ds)
        return self._flatten_predictions(pred_arr, index, context_df=df)

    def predict_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[float, np.ndarray]:
        """CV callback: fit on train, return horizon-1 quantile arrays aligned with test rows."""
        self.fit(train_df)
        full = pd.concat([train_df, test_df], ignore_index=True)
        full = full.drop_duplicates(subset=[GROUP_COL, TIME_IDX_COL], keep="last")
        max_train_idx = int(train_df[TIME_IDX_COL].max())
        min_pred = int(test_df[TIME_IDX_COL].min()) - 1
        preds = self.predict_quantiles(
            full,
            max_train_time_idx=max_train_idx,
            min_prediction_idx=min_pred,
        )
        h1 = preds[preds["horizon"] == 1].copy()
        merged = test_df[[GROUP_COL, TIMESTAMP_COL]].merge(
            h1,
            on=[GROUP_COL, TIMESTAMP_COL],
            how="left",
        )
        q_keys = {q: f"pred_q{int(q * 100):02d}" for q in self.quantiles}
        out: dict[float, np.ndarray] = {}
        for q, col in q_keys.items():
            if col not in merged.columns:
                out[q] = np.full(len(test_df), np.nan)
            else:
                out[q] = merged[col].to_numpy(dtype=float)
        return out

    def predict_fold_oof(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        fold: int,
    ) -> pd.DataFrame:
        """Full multi-horizon OOF rows for one CV fold."""
        self.fit(train_df)
        full = pd.concat([train_df, test_df], ignore_index=True)
        full = full.drop_duplicates(subset=[GROUP_COL, TIME_IDX_COL], keep="last")
        max_train_idx = int(train_df[TIME_IDX_COL].max())
        min_pred = int(test_df[TIME_IDX_COL].min()) - 1
        preds = self.predict_quantiles(
            full,
            max_train_time_idx=max_train_idx,
            min_prediction_idx=min_pred,
        )
        test_times = set(pd.to_datetime(test_df[TIMESTAMP_COL], utc=True))
        preds = preds[preds[TIMESTAMP_COL].isin(test_times)].copy()
        preds["fold"] = fold
        preds["model"] = "tft"
        preds["y"] = preds[TARGET_COL]
        return preds

    def latest_checkpoint(self) -> Path | None:
        """Return the newest checkpoint file if one exists."""
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return None
        ckpts = sorted(self.checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        return ckpts[-1] if ckpts else None

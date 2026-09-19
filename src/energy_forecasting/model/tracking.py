"""MLflow experiment tracking with optional Weights & Biases mirror."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow

from energy_forecasting.config import MLFLOW_EXPERIMENT, MLRUNS_DIR, WANDB_PROJECT

logger = logging.getLogger(__name__)

def _resolve_tracking_uri(uri: str | Path) -> str:
    """Return an MLflow tracking URI. Windows paths need a file:// scheme."""
    text = str(uri)
    if text.startswith(("http://", "https://", "file:", "sqlite:")):
        return text
    return Path(text).resolve().as_uri()


FOLD_METRIC_KEYS = (
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "crps",
    "pi_coverage",
    "mae",
    "rmse",
)


def _flatten_params(params: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            out[key] = ",".join(str(v) for v in value)
        else:
            out[key] = str(value)
    return out


class ExperimentTracker:
    """Log params, per-fold metrics, and artifacts to MLflow (and optionally W&B)."""

    def __init__(
        self,
        *,
        experiment_name: str = MLFLOW_EXPERIMENT,
        tracking_uri: str | Path | None = None,
        enable_wandb: bool = False,
        wandb_project: str = WANDB_PROJECT,
    ) -> None:
        self.experiment_name = experiment_name
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or MLRUNS_DIR
        self.tracking_uri = _resolve_tracking_uri(uri)
        self.enable_wandb = enable_wandb
        self.wandb_project = wandb_project
        self._wandb_run = None

    def start_run(self, run_name: str, params: Mapping[str, Any]) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=run_name)
        flat = _flatten_params(params)
        mlflow.log_params(flat)
        logger.info("MLflow run started: experiment=%s name=%s", self.experiment_name, run_name)

        if self.enable_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("W&B requested but wandb is not installed. pip install -e '.[wandb]'") from exc
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=dict(params),
                reinit=True,
            )
            logger.info("W&B run started: project=%s name=%s", self.wandb_project, run_name)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        if not numeric:
            return
        mlflow.log_metrics(numeric, step=step)
        if self._wandb_run is not None:
            payload = dict(numeric)
            if step is not None:
                payload["fold"] = step
            self._wandb_run.log(payload)

    def log_fold_metrics(self, fold: int, metrics: Mapping[str, float]) -> None:
        tagged = {}
        for key in FOLD_METRIC_KEYS:
            if key in metrics:
                tagged[f"fold_{fold}_{key}"] = float(metrics[key])
        self.log_metrics(tagged)
        self.log_metrics(
            {k: float(metrics[k]) for k in FOLD_METRIC_KEYS if k in metrics},
            step=fold,
        )

    def log_mean_metrics(self, fold_metrics) -> None:
        numeric = fold_metrics.select_dtypes(include="number")
        means = numeric.mean(numeric_only=True).to_dict()
        tagged = {f"mean_{k}": float(v) for k, v in means.items() if k in FOLD_METRIC_KEYS}
        self.log_metrics(tagged)

    def log_artifact(self, path: Path | str) -> None:
        artifact = Path(path)
        if not artifact.exists():
            logger.warning("skip artifact (missing): %s", artifact)
            return
        mlflow.log_artifact(str(artifact))
        if self._wandb_run is not None:
            self._wandb_run.save(str(artifact), policy="now")

    def end_run(self) -> None:
        mlflow.end_run()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        logger.info("tracking run ended")

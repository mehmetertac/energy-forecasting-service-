"""MLflow Model Registry helpers — register, promote, and load production models."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from energy_forecasting.config import MLFLOW_EXPERIMENT, MLRUNS_DIR, OOF_TFT_PARQUET
from energy_forecasting.model.tracking import _resolve_tracking_uri

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "tft-solar-quantile"
DEFAULT_SELECTION_METRIC = "mean_crps"
FORECAST_ARTIFACT_KEY = "forecast_table"
CHECKPOINT_ARTIFACT_KEY = "checkpoint"

FORECAST_COLUMNS = (
    "timestamp",
    "plant_id",
    "horizon",
    "pred_q10",
    "pred_q50",
    "pred_q90",
)


@dataclass(frozen=True)
class ProductionModelInfo:
    """Resolved production model identity — the API depends on this, not file paths."""

    name: str
    version: str
    stage: str
    run_id: str
    uri: str


@dataclass(frozen=True)
class RegisteredModelResult:
    """Outcome of registering and promoting a model version."""

    run_id: str
    model_name: str
    model_version: str
    stage: str
    uri: str


def _default_tracking_uri(tracking_uri: str | Path | None = None) -> str:
    """Resolve tracking URI: explicit arg, then MLFLOW_TRACKING_URI, then MLRUNS_DIR."""
    if tracking_uri is not None:
        return _resolve_tracking_uri(tracking_uri)
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return _resolve_tracking_uri(env_uri)
    return _resolve_tracking_uri(MLRUNS_DIR)


def _client(tracking_uri: str | Path | None = None) -> MlflowClient:
    uri = _default_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def select_best_run(
    *,
    metric: str = DEFAULT_SELECTION_METRIC,
    experiment_name: str = MLFLOW_EXPERIMENT,
    tracking_uri: str | Path | None = None,
) -> str:
    """Return the run ID with the lowest ``metric`` in the experiment."""
    uri = _default_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"no finished runs in experiment {experiment_name!r} — train first with scripts/train.py"
        )
    metric_col = metric if metric.startswith("metrics.") else f"metrics.{metric}"
    if metric_col not in runs.columns or pd.isna(runs.iloc[0][metric_col]):
        raise RuntimeError(f"best run missing metric {metric!r}")

    run_id = str(runs.iloc[0]["run_id"])
    run_name = runs.iloc[0].get("tags.mlflow.runName", run_id)
    logger.info(
        "selected best run: id=%s name=%s %s=%.4f",
        run_id,
        run_name,
        metric,
        float(runs.iloc[0][metric_col]),
    )
    return run_id


def _find_run_artifact(client: MlflowClient, run_id: str, suffix: str) -> str:
    """Return artifact relative path ending with ``suffix`` under the run."""
    stack: list[str] = [""]
    while stack:
        prefix = stack.pop()
        for entry in client.list_artifacts(run_id, path=prefix):
            rel = entry.path
            if rel.endswith(suffix):
                return rel
            if entry.is_dir:
                stack.append(rel)
    raise FileNotFoundError(f"artifact *{suffix} not found under run {run_id}")


def _normalize_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns and "ds" in out.columns:
        out["timestamp"] = out["ds"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "plant_id" not in out.columns and "unique_id" in out.columns:
        out["plant_id"] = out["unique_id"]
    missing = [c for c in FORECAST_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"forecast table missing columns: {missing}")
    return out[list(FORECAST_COLUMNS)].sort_values(["plant_id", "timestamp", "horizon"]).reset_index(drop=True)


class CachedForecastModel(mlflow.pyfunc.PythonModel):
    """Serve precomputed quantile forecasts from a parquet table (batch inference)."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        forecast_path = context.artifacts[FORECAST_ARTIFACT_KEY]
        self.forecasts = _normalize_forecast_frame(pd.read_parquet(forecast_path))

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input):
        if isinstance(model_input, list):
            if not model_input:
                raise ValueError("model_input must not be empty")
            model_input = model_input[0]
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("model_input must be a DataFrame, dict, or list thereof")

        plant_id = str(model_input.iloc[0]["plant_id"])
        horizon = int(model_input.iloc[0].get("horizon", 24))
        as_of = model_input.iloc[0].get("as_of")

        rows = self.forecasts[self.forecasts["plant_id"] == plant_id].copy()
        if rows.empty:
            return pd.DataFrame(columns=list(FORECAST_COLUMNS))

        if as_of is not None and not pd.isna(as_of):
            as_of_ts = pd.Timestamp(as_of, tz="UTC")
            eligible = rows[rows["timestamp"] <= as_of_ts]
            if not eligible.empty:
                latest = eligible["timestamp"].max()
                origin = rows[rows["timestamp"] == latest]["timestamp"].iloc[0]
                target_times = pd.date_range(
                    origin + pd.Timedelta(hours=1),
                    periods=horizon,
                    freq="h",
                    tz="UTC",
                )
                rows = rows[rows["timestamp"].isin(target_times)]
            rows = rows[rows["horizon"] <= horizon]
        else:
            rows = rows[rows["horizon"] <= horizon]

        return rows.sort_values(["horizon", "timestamp"]).reset_index(drop=True)


def register_best_model(
    *,
    run_id: str | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    metric: str = DEFAULT_SELECTION_METRIC,
    experiment_name: str = MLFLOW_EXPERIMENT,
    tracking_uri: str | Path | None = None,
    promote: bool = True,
) -> RegisteredModelResult:
    """Package run artifacts as pyfunc, register, and promote Staging → Production."""
    client = _client(tracking_uri)
    chosen_run = run_id or select_best_run(
        metric=metric,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    forecast_rel = _find_run_artifact(client, chosen_run, ".parquet")
    checkpoint_rel: str | None = None
    try:
        checkpoint_rel = _find_run_artifact(client, chosen_run, ".ckpt")
    except FileNotFoundError:
        logger.warning("no checkpoint artifact on run %s — registering forecast table only", chosen_run)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        forecast_local = client.download_artifacts(chosen_run, forecast_rel, dst_path=str(tmp_path))
        artifacts: dict[str, str] = {FORECAST_ARTIFACT_KEY: forecast_local}
        if checkpoint_rel is not None:
            checkpoint_local = client.download_artifacts(chosen_run, checkpoint_rel, dst_path=str(tmp_path))
            artifacts[CHECKPOINT_ARTIFACT_KEY] = checkpoint_local

        with mlflow.start_run(run_id=chosen_run):
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=CachedForecastModel(),
                artifacts=artifacts,
                registered_model_name=model_name,
            )

    version = str(model_info.registered_model_version)
    uri = f"models:/{model_name}/{version}"

    if promote:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        stage = "Production"
        uri = f"models:/{model_name}/Production"
    else:
        stage = "None"

    logger.info("registered %s version %s at stage %s (run=%s)", model_name, version, stage, chosen_run)
    return RegisteredModelResult(
        run_id=chosen_run,
        model_name=model_name,
        model_version=version,
        stage=stage,
        uri=uri,
    )


def get_production_model(
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    stage: str = "Production",
    tracking_uri: str | Path | None = None,
) -> ProductionModelInfo:
    """Return metadata for the current model at ``stage`` (default Production)."""
    client = _client(tracking_uri)
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(
            f"no {stage!r} version for registered model {model_name!r} — run scripts/register_model.py"
        )
    latest = versions[0]
    uri = f"models:/{model_name}/{stage}"
    return ProductionModelInfo(
        name=model_name,
        version=str(latest.version),
        stage=stage,
        run_id=str(latest.run_id),
        uri=uri,
    )


def load_production_pyfunc(
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    stage: str = "Production",
    tracking_uri: str | Path | None = None,
) -> tuple[ProductionModelInfo, mlflow.pyfunc.PyFuncModel]:
    """Load the pyfunc model for the current production stage."""
    info = get_production_model(model_name=model_name, stage=stage, tracking_uri=tracking_uri)
    uri = _default_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    model = mlflow.pyfunc.load_model(info.uri)
    return info, model


def package_local_forecast(
    forecast_path: Path | None = None,
    checkpoint_path: Path | None = None,
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    tracking_uri: str | Path | None = None,
    run_name: str = "registry-unit-test",
    promote: bool = True,
) -> RegisteredModelResult:
    """Register from local paths (used by unit tests)."""
    uri = _default_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("registry-unit-test")

    forecast_src = forecast_path or OOF_TFT_PARQUET
    if not Path(forecast_src).exists():
        raise FileNotFoundError(f"forecast parquet not found: {forecast_src}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        forecast_copy = tmp_path / "forecasts.parquet"
        shutil.copy2(forecast_src, forecast_copy)
        artifacts: dict[str, str] = {FORECAST_ARTIFACT_KEY: str(forecast_copy)}
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt_copy = tmp_path / "model.ckpt"
            shutil.copy2(checkpoint_path, ckpt_copy)
            artifacts[CHECKPOINT_ARTIFACT_KEY] = str(ckpt_copy)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric(DEFAULT_SELECTION_METRIC, 0.0)
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=CachedForecastModel(),
                artifacts=artifacts,
                registered_model_name=model_name,
            )

    client = _client(tracking_uri)
    version = str(model_info.registered_model_version)
    if promote:
        client.transition_model_version_stage(model_name, version, "Staging", archive_existing_versions=False)
        client.transition_model_version_stage(model_name, version, "Production", archive_existing_versions=True)
        stage = "Production"
        result_uri = f"models:/{model_name}/Production"
    else:
        stage = "None"
        result_uri = f"models:/{model_name}/{version}"

    return RegisteredModelResult(
        run_id=model_info.run_id,
        model_name=model_name,
        model_version=version,
        stage=stage,
        uri=result_uri,
    )

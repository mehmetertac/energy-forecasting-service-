"""SQLite store for published day-ahead batch forecasts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from energy_forecasting.config import FORECAST_DB, FORECAST_MAX_AGE_HOURS
from energy_forecasting.model.registry import FORECAST_COLUMNS

FORECAST_ROW_COLUMNS = (*FORECAST_COLUMNS, "actual")


class ForecastStoreError(Exception):
    """Base error for forecast store operations."""


class EmptyForecastError(ForecastStoreError):
    """No published batch exists in the store."""


class StaleForecastError(ForecastStoreError):
    """Published batch is older than the configured freshness window."""


@dataclass(frozen=True)
class BatchMetadata:
    """Metadata for the current published batch."""

    model_name: str
    model_version: str
    model_stage: str
    run_id: str
    generated_at: datetime

    @property
    def age(self) -> timedelta:
        return datetime.now(tz=UTC) - self.generated_at


def _connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Create forecast store tables if they do not exist."""
    path = Path(db_path or FORECAST_DB)
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS batch_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_stage TEXT NOT NULL,
                run_id TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS forecasts (
                batch_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                plant_id TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                pred_q10 REAL NOT NULL,
                pred_q50 REAL NOT NULL,
                pred_q90 REAL NOT NULL,
                actual REAL,
                FOREIGN KEY (batch_id) REFERENCES batch_runs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_forecasts_plant_horizon
                ON forecasts (batch_id, plant_id, horizon);
            """
        )
        conn.commit()


def publish_batch(
    forecasts: pd.DataFrame,
    *,
    model_name: str,
    model_version: str,
    model_stage: str,
    run_id: str,
    generated_at: datetime | None = None,
    db_path: Path | str | None = None,
) -> BatchMetadata:
    """Atomically replace the current batch with ``forecasts``."""
    if forecasts.empty:
        raise ValueError("forecasts must not be empty")

    path = Path(db_path or FORECAST_DB)
    init_db(path)
    generated = generated_at or datetime.now(tz=UTC)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)
    generated_iso = generated.isoformat()

    work = forecasts.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True).map(
        lambda ts: ts.isoformat()
    )

    with _connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM forecasts")
        conn.execute("DELETE FROM batch_runs")
        cursor = conn.execute(
            """
            INSERT INTO batch_runs (model_name, model_version, model_stage, run_id, generated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (model_name, model_version, model_stage, run_id, generated_iso),
        )
        batch_id = cursor.lastrowid
        rows = []
        for _, row in work.iterrows():
            actual = None
            if "actual" in work.columns and pd.notna(row.get("actual")):
                actual = float(row["actual"])
            rows.append(
                (
                    batch_id,
                    str(row["timestamp"]),
                    str(row["plant_id"]),
                    int(row["horizon"]),
                    float(row["pred_q10"]),
                    float(row["pred_q50"]),
                    float(row["pred_q90"]),
                    actual,
                )
            )
        conn.executemany(
            """
            INSERT INTO forecasts
                (batch_id, timestamp, plant_id, horizon, pred_q10, pred_q50, pred_q90, actual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return BatchMetadata(
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage,
        run_id=run_id,
        generated_at=generated,
    )


def get_batch_metadata(
    *,
    db_path: Path | str | None = None,
    max_age_hours: float | None = None,
) -> BatchMetadata:
    """Return metadata for the latest batch, optionally enforcing freshness."""
    path = Path(db_path or FORECAST_DB)
    if not path.exists():
        raise EmptyForecastError(f"forecast store not found: {path}")

    with _connect(path) as conn:
        row = conn.execute(
            """
            SELECT model_name, model_version, model_stage, run_id, generated_at
            FROM batch_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        raise EmptyForecastError("no published batch in forecast store")

    generated_at = datetime.fromisoformat(str(row["generated_at"]))
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)

    meta = BatchMetadata(
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        model_stage=str(row["model_stage"]),
        run_id=str(row["run_id"]),
        generated_at=generated_at,
    )
    if max_age_hours is not None:
        limit = max_age_hours
        if limit != float("inf") and meta.age > timedelta(hours=limit):
            raise StaleForecastError(
                f"published batch is stale (age={meta.age}, max={timedelta(hours=limit)})"
            )
    elif meta.age > timedelta(hours=FORECAST_MAX_AGE_HOURS):
        raise StaleForecastError(
            f"published batch is stale (age={meta.age}, max={timedelta(hours=FORECAST_MAX_AGE_HOURS)})"
        )
    return meta


def list_plants(*, db_path: Path | str | None = None) -> list[str]:
    """Return sorted plant ids in the current batch."""
    meta = _latest_metadata_or_empty(db_path)
    if meta is None:
        return []

    path = Path(db_path or FORECAST_DB)
    with _connect(path) as conn:
        batch_id = conn.execute("SELECT id FROM batch_runs ORDER BY id DESC LIMIT 1").fetchone()
        if batch_id is None:
            return []
        rows = conn.execute(
            "SELECT DISTINCT plant_id FROM forecasts WHERE batch_id = ? ORDER BY plant_id",
            (batch_id["id"],),
        ).fetchall()
    return [str(row["plant_id"]) for row in rows]


def get_forecasts(
    *,
    plant_id: str,
    horizon: int,
    as_of: str | None = None,
    db_path: Path | str | None = None,
    max_age_hours: float | None = None,
) -> tuple[BatchMetadata, pd.DataFrame]:
    """Return forecast rows for ``plant_id`` up to ``horizon``."""
    meta = get_batch_metadata(db_path=db_path, max_age_hours=max_age_hours)
    path = Path(db_path or FORECAST_DB)

    with _connect(path) as conn:
        batch_id = conn.execute("SELECT id FROM batch_runs ORDER BY id DESC LIMIT 1").fetchone()
        if batch_id is None:
            raise EmptyForecastError("no published batch in forecast store")
        rows = conn.execute(
            """
            SELECT timestamp, plant_id, horizon, pred_q10, pred_q50, pred_q90, actual
            FROM forecasts
            WHERE batch_id = ? AND plant_id = ?
            ORDER BY horizon, timestamp
            """,
            (batch_id["id"], plant_id),
        ).fetchall()

    if not rows:
        return meta, pd.DataFrame(columns=list(FORECAST_ROW_COLUMNS))

    df = pd.DataFrame([dict(row) for row in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["origin"] = df["timestamp"] - pd.to_timedelta(df["horizon"], unit="h")

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of, tz="UTC")
        eligible = df[df["timestamp"] <= as_of_ts]
        if not eligible.empty:
            latest_origin = eligible["origin"].max()
            df = df[df["origin"] == latest_origin]
    else:
        latest_origin = df["origin"].max()
        df = df[df["origin"] == latest_origin]

    df = df[df["horizon"] <= horizon].drop(columns=["origin"])
    return meta, df.sort_values(["horizon", "timestamp"]).reset_index(drop=True)


def _latest_metadata_or_empty(db_path: Path | str | None) -> BatchMetadata | None:
    path = Path(db_path or FORECAST_DB)
    if not path.exists():
        return None
    try:
        return get_batch_metadata(db_path=path, max_age_hours=float("inf"))
    except EmptyForecastError:
        return None

"""Orchestrate dataset fetch, disaggregation, validation, and persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from energy_forecasting.config import PROCESSED_DIR, RAW_DIR, REGION_ID
from energy_forecasting.data.coherence import verify_coherence
from energy_forecasting.data.disaggregate import disaggregate_regional
from energy_forecasting.data.plants import select_plants
from energy_forecasting.data.sources.open_meteo import fetch_all_plant_weather
from energy_forecasting.data.sources.opsd import (
    fetch_renewable_plants,
    fetch_time_series,
    load_regional_solar,
    load_solar_plants,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetBuildResult:
    plants_metadata: pd.DataFrame
    plants_hourly: pd.DataFrame
    region_hourly: pd.DataFrame
    coherence: dict
    validation: dict


def _validate_dataset(
    plants: pd.DataFrame,
    plants_hourly: pd.DataFrame,
    region_hourly: pd.DataFrame,
) -> dict:
    """Run post-assembly validation checks."""
    n_plants = plants["plant_id"].nunique()
    n_timestamps = region_hourly["timestamp"].nunique()
    over_capacity = plants_hourly[
        plants_hourly["power_mw"] > plants_hourly["capacity_mw"] + 1e-6
    ]
    negative = plants_hourly[plants_hourly["power_mw"] < -1e-6]
    missing_weather = plants_hourly[
        plants_hourly[["ghi_wm2", "temp_c", "cloud_cover_pct"]].isna().any(axis=1)
    ]

    return {
        "n_plants": n_plants,
        "n_timestamps": n_timestamps,
        "date_start": str(region_hourly["timestamp"].min()),
        "date_end": str(region_hourly["timestamp"].max()),
        "over_capacity_rows": len(over_capacity),
        "negative_power_rows": len(negative),
        "missing_weather_rows": len(missing_weather),
    }


def build_dataset(force_download: bool = False, force_process: bool = False) -> DatasetBuildResult:
    """
    Fetch raw data, disaggregate regional solar to plants, validate, and write parquet.

    Skips re-download if raw files exist unless force_download=True.
    Rebuilds processed outputs when force_process=True or processed files missing.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_files = [
        PROCESSED_DIR / "plants_metadata.parquet",
        PROCESSED_DIR / "plants_hourly.parquet",
        PROCESSED_DIR / "region_hourly.parquet",
    ]
    if not force_process and all(path.exists() for path in processed_files):
        logger.info("Processed dataset already exists; loading from disk.")
        from energy_forecasting.data.load import (
            load_plants_hourly,
            load_plants_metadata,
            load_region_hourly,
        )

        plants_metadata = load_plants_metadata()
        plants_hourly = load_plants_hourly()
        region_hourly = load_region_hourly()
        coherence = verify_coherence(plants_hourly, region_hourly)
        validation = _validate_dataset(plants_metadata, plants_hourly, region_hourly)
        return DatasetBuildResult(
            plants_metadata=plants_metadata,
            plants_hourly=plants_hourly,
            region_hourly=region_hourly,
            coherence=coherence,
            validation=validation,
        )

    fetch_time_series(force=force_download)
    fetch_renewable_plants(force=force_download)

    region = load_regional_solar()
    solar_plants = load_solar_plants()
    plants = select_plants(solar_plants)
    weather = fetch_all_plant_weather(plants, force=force_download)
    plants_hourly = disaggregate_regional(region, plants, weather)

    region_hourly = region.copy()
    region_hourly["region_id"] = REGION_ID

    coherence = verify_coherence(plants_hourly, region_hourly)
    validation = _validate_dataset(plants, plants_hourly, region_hourly)

    if not coherence["coherent"]:
        raise RuntimeError(
            f"Plant-region incoherent: max abs diff = {coherence['max_abs_diff_mw']:.6f} MW"
        )

    plants_metadata = plants[
        ["plant_id", "name", "capacity_mw", "lat", "lon", "region_id", "federal_state", "subregion_id"]
    ].copy()
    plants_metadata.to_parquet(PROCESSED_DIR / "plants_metadata.parquet", index=False)
    plants_hourly.to_parquet(PROCESSED_DIR / "plants_hourly.parquet", index=False)
    region_hourly.to_parquet(PROCESSED_DIR / "region_hourly.parquet", index=False)

    logger.info(
        "Dataset built: %d plants, %d hourly timestamps, max coherence diff %.2e MW",
        validation["n_plants"],
        validation["n_timestamps"],
        coherence["max_abs_diff_mw"],
    )

    return DatasetBuildResult(
        plants_metadata=plants_metadata,
        plants_hourly=plants_hourly,
        region_hourly=region_hourly,
        coherence=coherence,
        validation=validation,
    )

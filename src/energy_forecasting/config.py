"""Project-wide paths and default forecasting settings."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MLRUNS_DIR = ROOT_DIR / "mlruns"

DEFAULT_HORIZON = 24
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
DEFAULT_FREQ = "h"
DEFAULT_TZ = "Europe/Berlin"
TIMESTAMP_COL = "timestamp"
TARGET_COL = "power_mw"

MLFLOW_EXPERIMENT = "energy-forecasting-tft"
WANDB_PROJECT = "energy-forecasting-service"

METRICS_TFT_PLANT_CSV = ARTIFACTS_DIR / "metrics_tft_plant.csv"
OOF_TFT_PARQUET = ARTIFACTS_DIR / "oof_tft.parquet"
TFT_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "tft"
QUANTILE_BAND_PLOT = ARTIFACTS_DIR / "plots" / "quantile_bands.png"

FORECAST_DB = Path(os.environ.get("FORECAST_DB", str(ARTIFACTS_DIR / "forecasts.db")))
MODEL_MAX_AGE_HOURS = float(os.environ.get("MODEL_MAX_AGE_HOURS", "24"))
FORECAST_MAX_AGE_HOURS = float(os.environ.get("FORECAST_MAX_AGE_HOURS", "36"))

CV_N_SPLITS = 5

TFT_ENCODER_LENGTH = 168
TFT_MIN_ENCODER_LENGTH = 48
TFT_PREDICTION_LENGTH = 24
TFT_HIDDEN_SIZE = 64
TFT_ATTENTION_HEAD_SIZE = 4
TFT_DROPOUT = 0.1
TFT_LEARNING_RATE = 1e-3
TFT_MAX_EPOCHS = 30
TFT_BATCH_SIZE = 64
TFT_EARLY_STOPPING_PATIENCE = 5

DATA_START = "2018-01-01"
DATA_END = "2020-09-30"
REGION_ID = "DE_LU_PV"
N_PLANTS = 12
MIN_PLANT_CAPACITY_MW = 5.0

OPSD_TIME_SERIES_URL = (
    "https://data.open-power-system-data.org/time_series/"
    "2020-10-06/time_series_60min_singleindex.csv"
)
OPSD_RENEWABLE_PLANTS_URL = (
    "https://data.open-power-system-data.org/renewable_power_plants/"
    "2020-08-25/renewable_power_plants_DE.csv"
)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_REQUEST_DELAY_S = 0.5

REGIONAL_SOLAR_COLUMNS = (
    "DE_LU_solar_generation_actual",
    "DE_solar_generation_actual",
)

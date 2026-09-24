"""Publish day-ahead P10/P50/P90 forecasts from Production model to SQLite."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from energy_forecasting.config import DEFAULT_HORIZON, FORECAST_DB, MODEL_MAX_AGE_HOURS
from energy_forecasting.serving.batch import publish_day_ahead_forecasts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=FORECAST_DB,
        help=f"SQLite forecast store path (default: {FORECAST_DB})",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional features parquet — validates known-future covariate columns",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Max forecast horizon in hours (default: 24)",
    )
    parser.add_argument(
        "--model-max-age-hours",
        type=float,
        default=MODEL_MAX_AGE_HOURS,
        help="Refuse stale Production model versions older than this many hours",
    )
    parser.add_argument(
        "--seed-if-missing",
        action="store_true",
        help="Seed synthetic Production model when registry is empty (compose smoke only)",
    )
    args = parser.parse_args()

    try:
        meta = publish_day_ahead_forecasts(
            db_path=args.db,
            features_path=args.features,
            max_horizon=args.horizon,
            model_max_age_hours=args.model_max_age_hours,
            seed_if_missing=args.seed_if_missing,
        )
    except Exception as exc:
        logger.error("daily forecast job failed: %s", exc)
        return 1

    logger.info(
        "published %s v%s (%s) — %d rows to %s",
        meta.model_name,
        meta.model_version,
        meta.model_stage,
        args.horizon,
        args.db,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

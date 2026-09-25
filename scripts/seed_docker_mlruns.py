"""Seed a local MLflow store with tft-solar-quantile Production for Docker smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from energy_forecasting.serving.seed import seed_production_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("docker/mlruns-seed"),
        help="MLflow file store directory to create (default: docker/mlruns-seed)",
    )
    args = parser.parse_args()

    uri = seed_production_model(store=args.store)
    print(f"seeded {uri} at {args.store.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Register the best TFT MLflow run and promote it to Production."""

from __future__ import annotations

import argparse
import logging
import sys

from energy_forecasting.model.registry import (
    DEFAULT_SELECTION_METRIC,
    REGISTERED_MODEL_NAME,
    register_best_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register best TFT run in MLflow Model Registry.")
    parser.add_argument("--run-id", type=str, default=None, help="Override auto-selected best run.")
    parser.add_argument("--metric", type=str, default=DEFAULT_SELECTION_METRIC, help="Metric to minimize.")
    parser.add_argument("--model-name", type=str, default=REGISTERED_MODEL_NAME)
    parser.add_argument("--no-promote", action="store_true", help="Register only; skip Staging/Production.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = register_best_model(
        run_id=args.run_id,
        model_name=args.model_name,
        metric=args.metric,
        promote=not args.no_promote,
    )
    print("Model registered.")
    print(f"  Run ID:    {result.run_id}")
    print(f"  Name:      {result.model_name}")
    print(f"  Version:   {result.model_version}")
    print(f"  Stage:     {result.stage}")
    print(f"  URI:       {result.uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

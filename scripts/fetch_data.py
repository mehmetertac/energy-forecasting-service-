#!/usr/bin/env python
"""CLI: fetch OPSD/Open-Meteo and assemble plant hourly series."""

from __future__ import annotations

import argparse
import logging
import sys

from energy_forecasting.data.assemble import build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch OPSD/Open-Meteo data and assemble plant + regional hourly series."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw OPSD and Open-Meteo files even if cached.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of processed parquet outputs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = build_dataset(force_download=args.force_download, force_process=args.force)
    print("Dataset assembly complete.")
    print(f"  Plants:     {result.validation['n_plants']}")
    print(f"  Timestamps: {result.validation['n_timestamps']}")
    print(f"  Range:      {result.validation['date_start']} -> {result.validation['date_end']}")
    print(
        f"  Coherent:   {result.coherence['coherent']} "
        f"(max diff {result.coherence['max_abs_diff_mw']:.2e} MW)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

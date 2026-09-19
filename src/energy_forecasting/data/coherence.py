"""Plant-sum vs regional total coherence checks."""

from __future__ import annotations

import pandas as pd


def verify_coherence(
    plants_hourly: pd.DataFrame,
    region_hourly: pd.DataFrame,
    tolerance: float = 1e-6,
) -> dict:
    """Verify plant sums match regional totals at each timestamp."""
    plant_sums = (
        plants_hourly.groupby("timestamp", as_index=False)["power_mw"]
        .sum()
        .rename(columns={"power_mw": "plant_sum_mw"})
    )
    merged = plant_sums.merge(region_hourly, on="timestamp", how="inner")
    merged["abs_diff"] = (merged["plant_sum_mw"] - merged["power_mw"]).abs()
    max_diff = float(merged["abs_diff"].max()) if not merged.empty else float("nan")
    coherent = bool(merged.empty is False and max_diff <= tolerance)
    return {
        "coherent": coherent,
        "max_abs_diff_mw": max_diff,
        "n_timestamps": len(merged),
    }

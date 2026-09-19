"""Training artifact plots (quantile bands)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from energy_forecasting.config import QUANTILE_BAND_PLOT, TARGET_COL, TIMESTAMP_COL


def plot_quantile_bands(
    oof: pd.DataFrame,
    path: Path | str | None = None,
    *,
    plant_id: str | None = None,
    horizon: int = 1,
) -> Path:
    """Plot P10–P90 band, P50, and actuals for one plant at a single horizon."""
    work = oof.copy()
    if "horizon" in work.columns:
        work = work[work["horizon"] == horizon]
    if plant_id is None:
        plant_id = str(work["plant_id"].iloc[0])
    work = work[work["plant_id"].astype(str) == str(plant_id)].sort_values(TIMESTAMP_COL)

    out_path = Path(path or QUANTILE_BAND_PLOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ts = pd.to_datetime(work[TIMESTAMP_COL])
    if "pred_q10" in work.columns and "pred_q90" in work.columns:
        ax.fill_between(ts, work["pred_q10"], work["pred_q90"], alpha=0.25, label="P10–P90")
    if "pred_q50" in work.columns:
        ax.plot(ts, work["pred_q50"], label="P50", linewidth=1.5)
    y_col = TARGET_COL if TARGET_COL in work.columns else "y"
    if y_col in work.columns:
        ax.plot(ts, work[y_col], label="actual", linewidth=1.0, color="black")
    ax.set_title(f"TFT quantile bands — {plant_id} (h={horizon})")
    ax.set_ylabel("MW")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

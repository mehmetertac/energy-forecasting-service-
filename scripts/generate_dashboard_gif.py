"""Generate docs/dashboard.gif — P50 line with P10–P90 band over actuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "docs" / "dashboard.gif"
OUT_PNG = ROOT / "docs" / "dashboard.png"


def _solar_curve(hours: np.ndarray) -> np.ndarray:
    """Bell-shaped daytime generation (MW)."""
    return np.clip(520 * np.exp(-0.5 * ((hours - 12) / 3.2) ** 2), 0, None)


def _frame(*, horizon: int, dpi: int = 100) -> Image.Image:
    hours = np.arange(1, horizon + 1, dtype=float)
    timestamps = 6 + hours  # 06:00 UTC start

    actual = _solar_curve(timestamps)
    p50 = actual * (0.98 + 0.04 * np.sin(hours / 4))
    spread = 40 + 0.18 * p50
    p10 = np.clip(p50 - spread, 0, None)
    p90 = p50 + spread * 1.1

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.fill_between(hours, p10, p90, alpha=0.25, color="#4C78A8", label="P10–P90")
    ax.plot(hours, p50, color="#4C78A8", linewidth=2, label="P50")
    ax.plot(hours, actual, color="black", linewidth=1.5, label="Actual")

    ax.set_xlim(1, 24)
    ax.set_ylim(0, max(p90.max(), actual.max()) * 1.08)
    ax.set_xlabel("Target hour (UTC)")
    ax.set_ylabel("MW")
    ax.set_title(f"DE_PV_001 — latest origin, h=1…{horizon}")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Sidebar-style metrics strip (matches Streamlit layout)
    fig.text(
        0.02,
        0.92,
        "Model: tft-solar-quantile  |  Version: 1  |  Stage: Production  |  "
        "Pinball q50: 0.210  |  PI coverage: 78.5%",
        fontsize=9,
        color="#444444",
        va="top",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf)


def main() -> None:
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    horizons = [6, 12, 18, 24, 18, 12, 8, 24]
    frames = [_frame(horizon=h) for h in horizons]
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )

    # Full-width still for fallback / social previews
    full = _frame(horizon=24, dpi=120)
    full.save(OUT_PNG, optimize=True)

    print(f"wrote {OUT_GIF} ({OUT_GIF.stat().st_size // 1024} KB)")
    print(f"wrote {OUT_PNG} ({OUT_PNG.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

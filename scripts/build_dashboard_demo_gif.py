"""Build docs/dashboard.gif from live API (Streamlit-style chart, horizon scrub)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "docs" / "dashboard.gif"
API = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
PLANT = os.environ.get("DEMO_PLANT_ID", "DE_PV_001")


def _fetch(horizon: int) -> dict:
    body = json.dumps({"plant_id": PLANT, "horizon": horizon}).encode()
    req = urllib.request.Request(
        f"{API}/forecast",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _frame(rows: list[dict], *, horizon: int, dpi: int = 100) -> Image.Image:
    hours = np.array([r["horizon"] for r in rows], dtype=float)
    p50 = np.array([r["pred_q50"] for r in rows], dtype=float)
    p10 = np.array([r["pred_q10"] for r in rows], dtype=float)
    p90 = np.array([r["pred_q90"] for r in rows], dtype=float)
    actual = np.array([r["actual"] if r["actual"] is not None else np.nan for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.fill_between(hours, p10, p90, alpha=0.25, color="#4C78A8", label="P10–P90")
    ax.plot(hours, p50, color="#4C78A8", linewidth=2, label="P50")
    if np.isfinite(actual).any():
        ax.plot(hours, actual, color="black", linewidth=1.5, label="Actual")
    ax.set_xlim(1, 24)
    ax.set_ylim(0, max(p90.max(), np.nanmax(actual) if np.isfinite(actual).any() else 0) * 1.08)
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel("MW")
    ax.set_title(f"{PLANT} — forecast horizon scrub (h=1…{horizon})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


def main() -> int:
    try:
        payload = _fetch(24)
    except urllib.error.URLError as exc:
        print(f"API not reachable at {API}: {exc}", file=sys.stderr)
        return 1
    all_rows = sorted(payload["forecasts"], key=lambda r: r["horizon"])
    frames: list[Image.Image] = []
    for h in range(6, 25, 3):
        subset = [r for r in all_rows if r["horizon"] <= h]
        if subset:
            frames.append(_frame(subset, horizon=h))
    if not frames:
        print("no forecast rows", file=sys.stderr)
        return 1
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUT_GIF} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

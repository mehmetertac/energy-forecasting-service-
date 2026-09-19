"""Select diverse solar plants from OPSD registry."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from energy_forecasting.config import MIN_PLANT_CAPACITY_MW, N_PLANTS, REGION_ID


def slugify_subregion(name: str) -> str:
    """Map federal state name to a stable subregion identifier."""
    normalized = name.strip().upper()
    normalized = normalized.replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE").replace("ß", "SS")
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")
    if not normalized:
        return "DE_SR_UNKNOWN"
    return f"DE_SR_{normalized}"


def add_subregion_ids(plants: pd.DataFrame) -> pd.DataFrame:
    """Attach ``subregion_id`` from ``federal_state`` when present."""
    out = plants.copy()
    if "federal_state" not in out.columns:
        out["subregion_id"] = "DE_SR_UNKNOWN"
        return out
    out["subregion_id"] = out["federal_state"].fillna("Unknown").map(slugify_subregion)
    return out


def _greedy_spatial_sample(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick n plants spread across lat/lon using farthest-point sampling."""
    if len(candidates) <= n:
        return candidates.copy()

    coords = candidates[["lat", "lon"]].to_numpy()
    selected_idx = [int(candidates["capacity_mw"].idxmax())]
    remaining = set(candidates.index) - {selected_idx[0]}

    while len(selected_idx) < n and remaining:
        selected_coords = coords[[candidates.index.get_loc(i) for i in selected_idx]]
        best_idx = None
        best_dist = -1.0
        for idx in remaining:
            pos = candidates.index.get_loc(idx)
            point = coords[pos]
            min_dist = np.min(np.linalg.norm(selected_coords - point, axis=1))
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = idx
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return candidates.loc[selected_idx].copy()


def select_plants(
    solar_plants: pd.DataFrame,
    n_plants: int = N_PLANTS,
    min_capacity_mw: float = MIN_PLANT_CAPACITY_MW,
) -> pd.DataFrame:
    """Select n diverse utility-scale solar plants with stable IDs."""
    eligible = solar_plants[solar_plants["capacity_mw"] >= min_capacity_mw].copy()
    if len(eligible) < n_plants:
        raise ValueError(
            f"Only {len(eligible)} plants meet capacity >= {min_capacity_mw} MW; "
            f"need at least {n_plants}."
        )

    eligible = eligible.sort_values("capacity_mw", ascending=False).reset_index(drop=True)
    selected = _greedy_spatial_sample(eligible, n_plants)
    selected = selected.sort_values(["federal_state", "capacity_mw"], ascending=[True, False])
    selected = selected.reset_index(drop=True)
    selected["plant_id"] = [f"DE_PV_{i + 1:03d}" for i in range(len(selected))]
    selected["region_id"] = REGION_ID
    selected = add_subregion_ids(selected)

    if "name" not in selected.columns or selected["name"].isna().all():
        selected["name"] = selected["municipality"].fillna("Unknown")

    return selected[
        ["plant_id", "name", "capacity_mw", "lat", "lon", "region_id", "federal_state", "subregion_id"]
    ].reset_index(drop=True)

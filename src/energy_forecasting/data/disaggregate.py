"""Disaggregate regional solar generation to plant level."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clearsky_ghi(timestamp: pd.Series, lat: float, lon: float) -> np.ndarray:
    """Simple extraterrestrial + elevation clearsky GHI proxy (W/m²)."""
    ts = pd.to_datetime(timestamp, utc=True)
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=float)
    hour = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0

    declination = np.radians(23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365)))
    hour_angle = np.radians(15 * (hour - 12) + lon)
    lat_rad = np.radians(lat)

    sin_elevation = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(
        declination
    ) * np.cos(hour_angle)
    elevation = np.arcsin(np.clip(sin_elevation, -1, 1))
    clearsky = np.maximum(0.0, 1361.0 * np.sin(elevation) * 0.7)
    return clearsky


def compute_capacity_factors(
    weather: pd.DataFrame,
    plants: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-plant hourly capacity factors from weather."""
    plant_lookup = plants.set_index("plant_id")
    cf_frames = []

    for plant_id, group in weather.groupby("plant_id"):
        lat = float(plant_lookup.loc[plant_id, "lat"])
        lon = float(plant_lookup.loc[plant_id, "lon"])
        clearsky = _clearsky_ghi(group["timestamp"], lat, lon)
        ghi = group["ghi_wm2"].to_numpy(dtype=float)
        cf = np.divide(
            ghi,
            np.maximum(clearsky, 1.0),
            out=np.zeros_like(ghi),
            where=np.maximum(clearsky, 1.0) > 0,
        )
        cf = np.clip(cf, 0.0, 1.0)
        frame = group[["timestamp", "plant_id"]].copy()
        frame["capacity_factor"] = cf
        cf_frames.append(frame)

    return pd.concat(cf_frames, ignore_index=True)


def disaggregate_regional(
    region: pd.DataFrame,
    plants: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scale plant shares to match regional MW each hour.

    power_i(t) = region(t) * (capacity_i * cf_i(t)) / sum_j(capacity_j * cf_j(t))
    """
    capacity_factors = compute_capacity_factors(weather, plants)
    weights = capacity_factors.merge(
        plants[["plant_id", "capacity_mw"]],
        on="plant_id",
        how="left",
    )
    weights["weight"] = weights["capacity_mw"] * weights["capacity_factor"]

    region_indexed = region.set_index("timestamp")["power_mw"]
    weight_matrix = weights.pivot(
        index="timestamp",
        columns="plant_id",
        values="weight",
    ).reindex(region_indexed.index, fill_value=0.0)

    weight_sums = weight_matrix.sum(axis=1)
    regional_mw = region_indexed.reindex(weight_matrix.index, fill_value=0.0)

    capacity_weights = pd.Series(
        plants.set_index("plant_id")["capacity_mw"],
        dtype=float,
    )
    capacity_shares = capacity_weights / capacity_weights.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        shares = weight_matrix.div(weight_sums.replace(0, np.nan), axis=0)
        zero_weight_mask = weight_sums <= 0
        shares.loc[zero_weight_mask] = capacity_shares.to_numpy()
        shares = shares.fillna(0.0)

    plant_power = shares.mul(regional_mw, axis=0)
    plant_power.loc[regional_mw <= 0] = 0.0

    plant_power_long = (
        plant_power.reset_index()
        .melt(id_vars="timestamp", var_name="plant_id", value_name="power_mw")
    )

    merged = plant_power_long.merge(weather, on=["timestamp", "plant_id"], how="left")
    merged = merged.merge(
        plants[["plant_id", "capacity_mw"]],
        on="plant_id",
        how="left",
    )
    return merged[
        [
            "timestamp",
            "plant_id",
            "power_mw",
            "capacity_mw",
            "ghi_wm2",
            "temp_c",
            "cloud_cover_pct",
        ]
    ].sort_values(["timestamp", "plant_id"]).reset_index(drop=True)

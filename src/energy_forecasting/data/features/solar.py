"""Solar position and clear-sky irradiance via pvlib."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pvlib import atmosphere, clearsky, irradiance, solarposition

ClearSkyModel = Literal["ineichen", "simplified_solis"]


def localize_to_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Interpret or convert timestamps to timezone ``tz`` for site-local solar geometry."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas.DatetimeIndex")
    if index.tz is None:
        return index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return index.tz_convert(tz)


def get_solarposition(
    time: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    *,
    altitude: float = 0.0,
    **kwargs: Any,
) -> pd.DataFrame:
    return solarposition.get_solarposition(time, latitude, longitude, altitude=altitude, **kwargs)


def get_clearsky_irradiance(
    time: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    *,
    model: ClearSkyModel = "ineichen",
    altitude: float = 0.0,
    solar_position: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    sol = (
        solar_position
        if solar_position is not None
        else get_solarposition(time, latitude, longitude, altitude=altitude)
    )

    if model == "ineichen":
        linke = clearsky.lookup_linke_turbidity(time, latitude, longitude)
        pressure = atmosphere.alt2pres(altitude)
        rel_am = atmosphere.get_relative_airmass(sol["apparent_zenith"])
        abs_am = atmosphere.get_absolute_airmass(rel_am, pressure)
        dni_extra = irradiance.get_extra_radiation(time)
        return clearsky.ineichen(
            sol["apparent_zenith"],
            abs_am,
            linke,
            altitude,
            dni_extra,
            **kwargs,
        )

    if model == "simplified_solis":
        dni_extra = irradiance.get_extra_radiation(time)
        return clearsky.simplified_solis(
            sol["apparent_elevation"],
            dni_extra=dni_extra,
            **kwargs,
        )

    raise ValueError(f"unknown clearsky model: {model!r}")


def add_solar_features(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    tz: str,
    *,
    clearsky_model: ClearSkyModel = "ineichen",
    altitude_m: float = 0.0,
) -> pd.DataFrame:
    """Append solar position and clear-sky irradiance columns with ``sun_*`` / ``cs_*`` prefixes."""
    times_local = localize_to_tz(df.index, tz)
    sol_full = get_solarposition(times_local, latitude, longitude, altitude=altitude_m)
    sol = sol_full[["apparent_zenith", "apparent_elevation", "azimuth"]].add_prefix("sun_")
    sol.index = df.index

    cs = get_clearsky_irradiance(
        times_local,
        latitude,
        longitude,
        model=clearsky_model,
        altitude=altitude_m,
        solar_position=sol_full,
    ).add_prefix("cs_")
    cs.index = df.index

    return pd.concat([df, sol, cs], axis=1)

"""Feature engineering orchestration for plant-level modeling tables."""

from __future__ import annotations

import pandas as pd

from energy_forecasting.config import DEFAULT_TZ, TARGET_COL, TIMESTAMP_COL
from energy_forecasting.data.features.calendar import make_calendar_features
from energy_forecasting.data.features.solar import add_solar_features
from energy_forecasting.data.features.weather import add_weather_features


def build_plant_feature_frame(
    plants_hourly: pd.DataFrame,
    plants_metadata: pd.DataFrame,
    *,
    tz: str = DEFAULT_TZ,
    country: str = "DE",
) -> pd.DataFrame:
    """Return long-format modeling table with calendar, solar, and weather features."""
    meta = plants_metadata.set_index("plant_id")
    frames: list[pd.DataFrame] = []

    for plant_id, group in plants_hourly.groupby("plant_id"):
        plant_meta = meta.loc[plant_id]
        work = group.sort_values(TIMESTAMP_COL).set_index(TIMESTAMP_COL)
        local_index = work.index.tz_convert(tz)

        cal = make_calendar_features(local_index, country=country)
        cal.index = work.index
        work = pd.concat([work, cal], axis=1)

        solar_block = add_solar_features(
            pd.DataFrame(index=work.index),
            latitude=float(plant_meta["lat"]),
            longitude=float(plant_meta["lon"]),
            tz=tz,
        )
        work = pd.concat([work, solar_block], axis=1)
        work = add_weather_features(work)
        work["plant_id"] = plant_id
        work["subregion_id"] = plant_meta["subregion_id"]
        work["region_id"] = plant_meta["region_id"]
        frames.append(work.reset_index(names=TIMESTAMP_COL))

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"power_mw": TARGET_COL})
    out[TIMESTAMP_COL] = pd.to_datetime(out[TIMESTAMP_COL], utc=True)
    return out.sort_values([TIMESTAMP_COL, "plant_id"]).reset_index(drop=True)


def default_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return default exogenous feature column names from a built feature frame."""
    exclude = {
        TIMESTAMP_COL,
        TARGET_COL,
        "plant_id",
        "region_id",
        "subregion_id",
        "power_mw",
    }
    numeric = frame.select_dtypes(include="number").columns
    return [col for col in numeric if col not in exclude]

"""Calendar and holiday features (adapted from Week 2 energy_features)."""

from __future__ import annotations

import warnings

import holidays
import numpy as np
import pandas as pd


def _normalize_country(code: str) -> str:
    return code.strip().upper()


def _basic_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour
    day_of_week = index.dayofweek
    month = index.month
    is_weekend = (day_of_week >= 5).astype(np.int8)
    iso = index.isocalendar()
    week_of_year = iso.week.astype(np.int16)

    return pd.DataFrame(
        {
            "hour": hour.astype(np.int16),
            "day_of_week": day_of_week.astype(np.int8),
            "month": month.astype(np.int8),
            "week_of_year": week_of_year,
            "is_weekend": is_weekend,
        },
        index=index,
    )


def _cyclical_hour_day_of_year(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy(dtype=float)
    angle_h = 2.0 * np.pi * hour / 24.0

    day = index.dayofyear.to_numpy(dtype=float)
    leap = np.asarray(index.is_leap_year, dtype=bool)
    days_in_year = np.where(leap, 366.0, 365.0)
    angle_doy = 2.0 * np.pi * (day - 1.0) / days_in_year

    return pd.DataFrame(
        {
            "sin_hour": np.sin(angle_h),
            "cos_hour": np.cos(angle_h),
            "sin_day_of_year": np.sin(angle_doy),
            "cos_day_of_year": np.cos(angle_doy),
        },
        index=index,
    )


def _holiday_calendar_for_years(
    country: str,
    subdiv: str | None,
    years: range | list[int],
) -> holidays.HolidayBase:
    ctry = _normalize_country(country)
    try:
        if subdiv:
            return holidays.country_holidays(ctry, subdiv=subdiv, years=years)
        return holidays.country_holidays(ctry, years=years)
    except (NotImplementedError, KeyError) as exc:
        raise ValueError(f"Unsupported country/subdivision: {country!r}/{subdiv!r}") from exc


def _holiday_dates_ns(
    country: str, subdiv: str | None, start: pd.Timestamp, end: pd.Timestamp
) -> np.ndarray:
    years = list(range(start.year, end.year + 1))
    cal = _holiday_calendar_for_years(country, subdiv, years)
    days = sorted(pd.Timestamp(d).normalize() for d in cal.keys())
    return np.array(days, dtype="datetime64[ns]")


def _holiday_distance_vectors(
    ord_dates: np.ndarray,
    holiday_ordinals_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if holiday_ordinals_sorted.size == 0:
        nan = np.full(len(ord_dates), np.nan)
        return nan, nan

    ord_h = holiday_ordinals_sorted
    idx_prev = np.searchsorted(ord_h, ord_dates, side="right") - 1
    idx_next = np.searchsorted(ord_h, ord_dates, side="left")

    prev_ord = np.where(idx_prev >= 0, ord_h[idx_prev], np.nan)
    next_ord = np.where(idx_next < len(ord_h), ord_h[idx_next], np.nan)

    days_since = np.where(np.isnan(prev_ord), np.nan, ord_dates - prev_ord).astype(float)
    days_to = np.where(np.isnan(next_ord), np.nan, next_ord - ord_dates).astype(float)
    return days_since, days_to


def _holiday_features(
    index: pd.DatetimeIndex,
    *,
    country: str,
    subdiv: str | None,
) -> pd.DataFrame:
    buf_start = index.min().normalize() - pd.Timedelta(days=400)
    buf_end = index.max().normalize() + pd.Timedelta(days=400)
    hol_ns = _holiday_dates_ns(country, subdiv, buf_start, buf_end)

    ord_dates = np.array([ts.normalize().date().toordinal() for ts in index], dtype=np.int64)
    holiday_ord = np.sort(
        np.unique(np.array([pd.Timestamp(ts).date().toordinal() for ts in hol_ns], dtype=np.int64))
    )
    days_since, days_to = _holiday_distance_vectors(ord_dates, holiday_ord)

    is_holiday = np.zeros(len(index), dtype=np.int8)
    if hol_ns.size:
        holiday_dates = {pd.Timestamp(ts).date() for ts in hol_ns}
        is_holiday = np.array([ts.date() in holiday_dates for ts in index], dtype=np.int8)

    return pd.DataFrame(
        {
            "is_holiday": is_holiday,
            "days_since_last_holiday": days_since,
            "days_to_next_holiday": days_to,
        },
        index=index,
    )


def make_calendar_features(
    index: pd.DatetimeIndex,
    *,
    country: str = "DE",
    subdiv: str | None = None,
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """Build calendar and holiday features aligned to ``index``."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"index must be DatetimeIndex, got {type(index)!r}")
    if index.tz is None:
        warnings.warn(
            "DatetimeIndex is timezone-naive; calendar columns derive local interpretations.",
            stacklevel=2,
        )

    basic = _basic_calendar_features(index)
    hol = _holiday_features(index, country=country, subdiv=subdiv)
    parts = [basic, hol]
    if include_cyclical:
        parts.append(_cyclical_hour_day_of_year(index))
    return pd.concat(parts, axis=1)

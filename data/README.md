# Data

Plant-level solar series used to train the TFT. **Do not commit raw OPSD CSVs or Open-Meteo JSON.**

## Sources

| Layer | Source | Notes |
|-------|--------|-------|
| Regional actuals | [OPSD time_series 2020-10-06](https://data.open-power-system-data.org/time_series/) | `DE_LU_solar_generation_actual` |
| Plant catalog | [OPSD renewable_power_plants DE](https://data.open-power-system-data.org/renewable_power_plants/) | 12 sites ≥ 5 MW, spatially sampled |
| Weather | [Open-Meteo Archive](https://open-meteo.com/) | GHI, temperature, cloud cover per plant |

Date range: 2018-01-01 → 2020-09-30 (hourly). Plant MW is **weather-weighted disaggregation** of the regional total, not metered plant telemetry.

## Build

```powershell
# Optional: copy Week 5 cache
Copy-Item -Recurse ..\multi-site-solar-hierarchy\data\raw data\raw
python scripts/fetch_data.py
```

Writes gitignored parquet:

- `data/processed/plants_metadata.parquet`
- `data/processed/plants_hourly.parquet` — `timestamp`, `plant_id`, `power_mw`, `capacity_mw`, `ghi_wm2`, `temp_c`, `cloud_cover_pct`
- `data/processed/region_hourly.parquet`

Assembly checks plant sums vs regional MW (`verify_coherence`).

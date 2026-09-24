"""Streamlit dashboard — P50 line with P10–P90 band over actuals."""

from __future__ import annotations

import altair as alt
import httpx
import streamlit as st
from client import (
    api_url,
    fetch_forecast,
    fetch_health,
    fetch_metrics,
    fetch_plants,
    filter_latest_origin,
)


def _fmt(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value:.1%}"
    return f"{value:.3f}"


st.set_page_config(page_title="Solar forecast dashboard", layout="wide")
st.title("Solar forecast dashboard")
st.caption(f"API: `{api_url()}`")

try:
    health = fetch_health()
except httpx.HTTPError as exc:
    st.error(f"Cannot reach API at {api_url()}: {exc}")
    st.stop()

if health.get("status") != "ok":
    st.warning(f"API unhealthy: {health.get('detail', 'unknown error')}")
    st.stop()

with st.sidebar:
    st.subheader("Controls")
    try:
        plants = fetch_plants()
    except httpx.HTTPError as exc:
        st.error(f"Failed to load plants: {exc}")
        st.stop()

    if not plants:
        st.warning("No plants in the production forecast cache.")
        st.stop()

    plant_id = st.selectbox("Site", plants)
    horizon = st.slider("Forecast horizon (hours)", min_value=1, max_value=24, value=24)

try:
    payload = fetch_forecast(plant_id=plant_id, horizon=horizon)
except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 404:
        st.error(f"No cached forecasts for {plant_id!r}.")
    else:
        st.error(f"Forecast request failed: {exc.response.text}")
    st.stop()
except httpx.HTTPError as exc:
    st.error(f"Forecast request failed: {exc}")
    st.stop()

chart_df = filter_latest_origin(payload["forecasts"], horizon=horizon)
if chart_df.empty:
    st.warning("No forecast rows for the selected horizon.")
    st.stop()

metrics_cols = st.columns(5)
metrics_cols[0].metric("Model", payload.get("model_name", "—"))
metrics_cols[1].metric("Version", payload.get("model_version", "—"))
metrics_cols[2].metric("Stage", payload.get("model_stage", "—"))

try:
    run_metrics = fetch_metrics()
    metrics_cols[3].metric("Pinball q50", _fmt(run_metrics.get("pinball_q50")))
    metrics_cols[4].metric("PI coverage", _fmt(run_metrics.get("pi_coverage"), pct=True))
except httpx.HTTPError:
    metrics_cols[3].metric("Pinball q50", "—")
    metrics_cols[4].metric("PI coverage", "—")

st.subheader(f"{plant_id} — latest origin, h=1…{horizon}")

base = alt.Chart(chart_df).encode(x=alt.X("timestamp:T", title="Target hour (UTC)"))

band = base.mark_area(opacity=0.25, color="#4C78A8").encode(
    y=alt.Y("pred_q10:Q", title="MW"),
    y2="pred_q90:Q",
)
p50 = base.mark_line(color="#4C78A8", strokeWidth=2).encode(y="pred_q50:Q")

layers: list[alt.Chart] = [band, p50]
if "actual" in chart_df.columns and chart_df["actual"].notna().any():
    actual = base.mark_line(color="black", strokeWidth=1.5).encode(y="actual:Q")
    layers.append(actual)

chart = alt.layer(*layers).resolve_scale(y="shared").properties(height=400)
st.altair_chart(chart, use_container_width=True)

with st.expander("Raw forecast rows"):
    st.dataframe(chart_df, use_container_width=True)

"""Coherence and plant-selection helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.data.coherence import verify_coherence
from energy_forecasting.data.plants import add_subregion_ids, slugify_subregion


def test_slugify_subregion():
    assert slugify_subregion("Niedersachsen") == "DE_SR_NIEDERSACHSEN"
    assert slugify_subregion("  ") == "DE_SR_UNKNOWN"


def test_add_subregion_ids():
    plants = pd.DataFrame({"federal_state": ["Bayern", None]})
    out = add_subregion_ids(plants)
    assert out["subregion_id"].iloc[0].startswith("DE_SR_")
    assert out["subregion_id"].iloc[1] == "DE_SR_UNKNOWN"


def test_verify_coherence_ok():
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    plants = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "plant_id": ["A", "B"],
            "power_mw": [3.0, 7.0],
        }
    )
    region = pd.DataFrame({"timestamp": [ts], "power_mw": [10.0]})
    result = verify_coherence(plants, region)
    assert result["coherent"] is True
    assert result["max_abs_diff_mw"] == pytest.approx(0.0)


def test_verify_coherence_fails():
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    plants = pd.DataFrame(
        {
            "timestamp": [ts],
            "plant_id": ["A"],
            "power_mw": [1.0],
        }
    )
    region = pd.DataFrame({"timestamp": [ts], "power_mw": [10.0]})
    result = verify_coherence(plants, region)
    assert result["coherent"] is False

"""Unit tests for rolling-origin CV splits."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.model.cv import RollingOriginSplit, rolling_origin_time_folds


def test_rolling_origin_expanding_folds_cover_later_times():
    times = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    folds = rolling_origin_time_folds(times, n_splits=4, gap=0)
    assert len(folds) >= 2
    assert folds[0].train_times.max() < folds[0].test_times.min()
    assert folds[-1].n_train >= folds[0].n_train


def test_rolling_origin_gap_prevents_overlap():
    times = pd.date_range("2020-01-01", periods=30, freq="h", tz="UTC")
    folds = rolling_origin_time_folds(times, n_splits=3, gap=2)
    for fold in folds:
        gap_times = set(fold.train_times).intersection(fold.test_times)
        assert not gap_times


def test_split_frame_yields_nonempty():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC"),
            "y": range(40),
        }
    )
    splitter = RollingOriginSplit(n_splits=3, gap=1, time_col="timestamp")
    pairs = list(splitter.split_frame(df))
    assert len(pairs) >= 2
    train, test, fold = pairs[0]
    assert not train.empty and not test.empty
    assert fold.fold == 1


def test_n_splits_must_be_at_least_two():
    times = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        rolling_origin_time_folds(times, n_splits=1)

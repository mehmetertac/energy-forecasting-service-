"""Rolling-origin cross-validation harness."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from energy_forecasting.config import TIMESTAMP_COL

SplitMode = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class TimeFold:
    train_times: pd.DatetimeIndex
    test_times: pd.DatetimeIndex
    fold: int

    @property
    def n_train(self) -> int:
        return len(self.train_times)

    @property
    def n_test(self) -> int:
        return len(self.test_times)


def unique_sorted_times(times: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(times).drop_duplicates().sort_values())


def rolling_origin_time_folds(
    times: pd.DatetimeIndex | pd.Series,
    n_splits: int = 5,
    *,
    mode: SplitMode = "expanding",
    max_train_size: int | None = None,
    test_size: int | None = None,
    gap: int = 0,
) -> list[TimeFold]:
    """Build expanding or rolling-origin folds on unique sorted timestamps."""
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    if mode == "rolling" and max_train_size is None:
        raise ValueError("max_train_size is required when mode='rolling'")

    unique = unique_sorted_times(times)
    n = len(unique)
    if n < n_splits + 1:
        raise ValueError(
            f"need at least {n_splits + 1} unique timestamps for {n_splits} folds, got {n}"
        )

    min_train = max(1, n // (n_splits + 1))
    span = test_size if test_size is not None else max(1, (n - min_train) // n_splits)

    folds: list[TimeFold] = []
    for k in range(n_splits):
        test_start = min_train + k * span
        test_end = test_start + span if k < n_splits - 1 else n
        if test_start >= n or test_start >= test_end:
            break

        train_end = test_start - gap
        if train_end < 1:
            continue

        if mode == "expanding":
            train = unique[:train_end]
        else:
            train_start = max(0, train_end - int(max_train_size))
            train = unique[train_start:train_end]

        test = unique[test_start:test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        folds.append(TimeFold(train_times=train, test_times=test, fold=k + 1))

    if not folds:
        raise ValueError("could not construct any rolling-origin folds")
    return folds


class RollingOriginSplit(BaseCrossValidator):
    """Sklearn-compatible rolling-origin splitter keyed by a timestamp column."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        mode: SplitMode = "expanding",
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
        time_col: str = TIMESTAMP_COL,
    ) -> None:
        self.n_splits = n_splits
        self.mode = mode
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None and groups is None:
            return self.n_splits
        return len(self._time_folds(self._resolve_times(X, groups)))

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        time_values = self._row_time_values(X, groups, n)
        folds = self._time_folds(unique_sorted_times(time_values))
        for fold in folds:
            train_mask = time_values.isin(fold.train_times).to_numpy()
            test_mask = time_values.isin(fold.test_times).to_numpy()
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx

    def split_times(self, times: pd.DatetimeIndex | pd.Series) -> Iterator[TimeFold]:
        yield from self._time_folds(unique_sorted_times(times))

    def split_frame(
        self,
        df: pd.DataFrame,
        *,
        time_col: str | None = None,
        sort: bool = True,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, TimeFold]]:
        col = time_col or self.time_col
        work = df.sort_values(col) if sort else df
        for fold in self.split_times(work[col]):
            train = work[work[col].isin(fold.train_times)]
            test = work[work[col].isin(fold.test_times)]
            if not train.empty and not test.empty:
                yield train, test, fold

    def _time_folds(self, times: pd.DatetimeIndex) -> list[TimeFold]:
        return rolling_origin_time_folds(
            times,
            self.n_splits,
            mode=self.mode,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap,
        )

    def _row_time_values(self, X, groups, n: int) -> pd.Series:
        if groups is not None:
            return pd.Series(groups, index=np.arange(n))
        if isinstance(X, pd.DataFrame) and self.time_col in X.columns:
            return pd.Series(X[self.time_col].to_numpy(), index=np.arange(n))
        raise ValueError(f"groups or DataFrame with {self.time_col!r} required")

    def _resolve_times(self, X, groups) -> pd.DatetimeIndex:
        n = len(X) if X is not None else 0
        return unique_sorted_times(self._row_time_values(X, groups, n))

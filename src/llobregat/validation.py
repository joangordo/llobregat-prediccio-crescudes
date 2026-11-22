"""Forward-chaining validation by hydrological year.

Each fold: training | gap | calibration | gap | test, with an inner validation (the last
years of the training period) to choose hyperparameters. The last years are the final
hold-out and are evaluated only once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from llobregat.config import ValidationConfig


def hydro_year(index: pd.DatetimeIndex, start_month: int = 10) -> np.ndarray:
    """Hydrological year: year 2020 runs from October 2019 to September 2020."""
    return np.where(index.month >= start_month, index.year + 1, index.year)


@dataclass
class Fold:
    name: str
    test_years: list[int]
    calib_years: list[int]
    train_years: list[int]
    inner_val_years: list[int]
    train: np.ndarray  # boolean masks over the index
    inner_train: np.ndarray
    inner_val: np.ndarray
    calib: np.ndarray
    test: np.ndarray

    def summary(self) -> dict:
        return {
            "fold": self.name,
            "train": f"{self.train_years[0]}–{self.train_years[-1]}",
            "inner_val": f"{self.inner_val_years[0]}–{self.inner_val_years[-1]}",
            "calib": ",".join(map(str, self.calib_years)),
            "test": ",".join(map(str, self.test_years)),
            "n_train": int(self.train.sum()),
            "n_calib": int(self.calib.sum()),
            "n_test": int(self.test.sum()),
        }


def _purge_before(
    mask: np.ndarray, index: pd.DatetimeIndex, boundary: pd.Timestamp, gap: pd.Timedelta
) -> np.ndarray:
    """Remove from `mask` the rows within the gap before `boundary`."""
    return mask & ~((index >= boundary - gap) & (index < boundary))


def make_folds(index: pd.DatetimeIndex, cfg: ValidationConfig) -> list[Fold]:
    """Outer folds + final fold (test = hold-out)."""
    years = hydro_year(index, cfg.hydro_year_start_month)
    uniq = sorted(np.unique(years))
    n_calib, n_inner, n_hold = cfg.calibration_years, cfg.inner_val_years, cfg.holdout_years
    min_train = n_inner + 1
    needed = min_train + n_calib + cfg.n_outer_folds + n_hold
    if len(uniq) < needed:
        raise ValueError(
            f"At least {needed} hydrological years are needed and there are {len(uniq)}"
        )

    gap = pd.Timedelta(days=cfg.gap_days)
    start_of = {y: index[years == y].min() for y in uniq}

    def build(name: str, test_years: list[int]) -> Fold:
        first_test = test_years[0]
        pos = uniq.index(first_test)
        calib_years = uniq[pos - n_calib : pos]
        train_years = uniq[: pos - n_calib]
        inner_val_years = train_years[-n_inner:]

        test = np.isin(years, test_years)
        calib = np.isin(years, calib_years)
        train = np.isin(years, train_years)
        # Gaps: training | gap | calibration | gap | test
        calib = _purge_before(calib, index, start_of[first_test], gap)
        train = _purge_before(train, index, start_of[calib_years[0]], gap)

        inner_val = np.isin(years, inner_val_years) & train
        inner_train = _purge_before(train & ~inner_val, index, start_of[inner_val_years[0]], gap)
        return Fold(
            name,
            test_years,
            calib_years,
            train_years,
            inner_val_years,
            train,
            inner_train,
            inner_val,
            calib,
            test,
        )

    holdout = uniq[-n_hold:] if n_hold else []
    eval_years = uniq[: len(uniq) - n_hold][-cfg.n_outer_folds :]
    folds = [build(f"fold_{i + 1}", [y]) for i, y in enumerate(eval_years)]
    if holdout:
        folds.append(build("final", holdout))
    return folds


def folds_table(folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame([f.summary() for f in folds])

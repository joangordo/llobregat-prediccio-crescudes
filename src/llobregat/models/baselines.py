"""Baselines: persistence and climatology, with empirical quantiles."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Persistence:
    """Q(t+h) = Q(t). The quantiles come from the historical error at horizon h."""

    def __init__(self, h_steps: int, quantiles=(0.1, 0.5, 0.9)):
        self.h_steps = h_steps
        self.quantiles = tuple(quantiles)
        self.residual_q: dict[float, float] = {}

    def fit(self, q: pd.Series) -> Persistence:
        resid = (q.shift(-self.h_steps) - q).dropna()
        self.residual_q = {tau: float(resid.quantile(tau)) for tau in self.quantiles}
        return self

    def predict(self, q_now) -> np.ndarray:
        return np.asarray(q_now, dtype=float)

    def predict_quantiles(self, q_now) -> dict[float, np.ndarray]:
        base = np.asarray(q_now, dtype=float)
        return {tau: np.clip(base + r, 0, None) for tau, r in self.residual_q.items()}


class Climatology:
    """Flow quantiles by day of year, with a window of ±window_days days."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9), window_days: int = 7):
        self.quantiles = tuple(quantiles)
        self.window_days = window_days
        self.table: pd.DataFrame | None = None

    def fit(self, q: pd.Series) -> Climatology:
        q = q.dropna()
        doy = q.index.day_of_year.to_numpy()
        values = q.to_numpy(dtype=float)
        rows = {}
        for d in range(1, 367):
            dist = np.abs(doy - d)
            dist = np.minimum(dist, 366 - dist)
            sel = values[dist <= self.window_days]
            rows[d] = [np.quantile(sel, t) if len(sel) else np.nan for t in self.quantiles]
        self.table = pd.DataFrame.from_dict(rows, orient="index", columns=list(self.quantiles))
        return self

    def predict_quantiles(
        self, index: pd.DatetimeIndex, h_steps: int = 0, resolution_min: int = 30
    ) -> dict[float, np.ndarray]:
        target_time = index + pd.Timedelta(minutes=h_steps * resolution_min)
        looked = self.table.loc[target_time.day_of_year]
        return {tau: looked[tau].to_numpy() for tau in self.quantiles}

    def predict(
        self, index: pd.DatetimeIndex, h_steps: int = 0, resolution_min: int = 30
    ) -> np.ndarray:
        return self.predict_quantiles(index, h_steps, resolution_min)[0.5]

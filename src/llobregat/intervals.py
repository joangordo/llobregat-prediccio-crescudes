"""Prediction interval building blocks: conformalised quantile regression (CQR) and ACI.

- CQR (Romano et al., 2019): calibrates a quantile interval [q_lo, q_hi] with a
  calibration set so that it has marginal coverage 1 − alpha.
- Mondrian: separate calibration per regime (e.g. normal / flood).
- ACI (Gibbs and Candès, 2021): online adjustment of the level to compensate for time
  series not being exchangeable.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def rearrange(*quantiles) -> list[np.ndarray]:
    """Monotone rearrangement: prevents quantile crossing (q_lo ≤ q_mid ≤ q_hi)."""
    stacked = np.sort(np.vstack([np.asarray(q, dtype=float) for q in quantiles]), axis=0)
    return list(stacked)


def conformal_quantile(scores, level: float) -> float:
    """Conformal quantile with finite-sample correction: s_(k), k = ⌈(n+1)·level⌉."""
    s = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * level)
    return math.inf if k > n else float(s[k - 1])


def cqr_scores(y, lo, hi) -> np.ndarray:
    """E_i = max(q_lo − y, y − q_hi): how far outside the interval (negative if inside)."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return np.maximum(lo - y, y - hi)


class CQR:
    """Conformalised quantile regression (symmetric or asymmetric)."""

    def __init__(self, alpha: float = 0.2, asymmetric: bool = False):
        self.alpha = alpha
        self.asymmetric = asymmetric
        self.q_lo_adj = self.q_hi_adj = None

    def fit(self, y_cal, lo_cal, hi_cal) -> CQR:
        y, lo, hi = (np.asarray(a, dtype=float) for a in (y_cal, lo_cal, hi_cal))
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        y, lo, hi = y[ok], lo[ok], hi[ok]
        if self.asymmetric:
            self.q_lo_adj = conformal_quantile(lo - y, 1 - self.alpha / 2)
            self.q_hi_adj = conformal_quantile(y - hi, 1 - self.alpha / 2)
        else:
            q = conformal_quantile(cqr_scores(y, lo, hi), 1 - self.alpha)
            self.q_lo_adj = self.q_hi_adj = q
        return self

    def predict(self, lo, hi) -> tuple[np.ndarray, np.ndarray]:
        if self.q_lo_adj is None:
            raise RuntimeError("Call fit() before predict()")
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        return lo - self.q_lo_adj, hi + self.q_hi_adj


class MondrianCQR:
    """CQR calibrated per regime. Regimes with too little data use the global one.

    The regime must be defined with information available at prediction time (e.g. the
    predicted median above the threshold), never with the observation.
    """

    def __init__(self, alpha: float = 0.2, min_n: int = 100, asymmetric: bool = False):
        self.alpha, self.min_n, self.asymmetric = alpha, min_n, asymmetric
        self.global_: CQR | None = None
        self.by_group: dict = {}

    def fit(self, y_cal, lo_cal, hi_cal, groups) -> MondrianCQR:
        y, lo, hi, g = (np.asarray(a) for a in (y_cal, lo_cal, hi_cal, groups))
        self.global_ = CQR(self.alpha, self.asymmetric).fit(y, lo, hi)
        self.by_group = {}
        for grp in np.unique(g):
            m = g == grp
            if m.sum() >= self.min_n:
                self.by_group[grp] = CQR(self.alpha, self.asymmetric).fit(y[m], lo[m], hi[m])
        return self

    def predict(self, lo, hi, groups) -> tuple[np.ndarray, np.ndarray]:
        lo, hi, g = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), np.asarray(groups)
        out_lo, out_hi = self.global_.predict(lo, hi)
        for grp, model in self.by_group.items():
            m = g == grp
            out_lo[m], out_hi[m] = model.predict(lo[m], hi[m])
        return out_lo, out_hi


class AdaptiveConformal:
    """ACI: alpha_{t+1} = alpha_t + gamma·(alpha − err_t).

    Initialised with the CQR calibration scores. The width at each time is the 1 − alpha_t
    quantile of the scores. If `window` is given, new scores are added and the `window`
    most recent ones are kept.
    In production, update() is called when the real value arrives (with delay = horizon).
    """

    def __init__(
        self, cal_scores, alpha: float = 0.2, gamma: float = 0.005, window: int | None = None
    ):
        self.alpha, self.gamma, self.window = alpha, gamma, window
        self.alpha_t = alpha
        self.scores = list(np.asarray(cal_scores, dtype=float))
        self.history: list[float] = []

    def current_q(self) -> float:
        a = min(max(self.alpha_t, 1e-6), 1 - 1e-6)
        if self.alpha_t <= 0:
            return float(np.max(self.scores))
        return conformal_quantile(self.scores, 1 - a)

    def interval(self, lo, hi) -> tuple[np.ndarray, np.ndarray]:
        q = self.current_q()
        return np.asarray(lo, dtype=float) - q, np.asarray(hi, dtype=float) + q

    def update(self, y: float, lo: float, hi: float) -> None:
        """Update with the real observation of an issued interval (lo, hi of the quantile model)."""
        q = self.current_q()
        err = float(not (lo - q <= y <= hi + q))
        self.alpha_t += self.gamma * (self.alpha - err)
        self.history.append(err)
        if self.window is not None:
            self.scores.append(float(max(lo - y, y - hi)))
            self.scores = self.scores[-self.window :]

    def long_run_error(self) -> float:
        return float(np.mean(self.history)) if self.history else math.nan


def regime_from_prediction(median_pred, threshold: float) -> np.ndarray:
    """Mondrian regime from the predicted median (available at prediction time)."""
    return np.where(np.asarray(median_pred, dtype=float) > threshold, "flood", "normal")


def alert_rule(upper, threshold: float) -> np.ndarray:
    """Warning if the upper bound of the interval exceeds the threshold."""
    return np.asarray(upper, dtype=float) > threshold


ScoreFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

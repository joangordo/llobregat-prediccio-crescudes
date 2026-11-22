"""Weighting schemes for the imbalance between normal flows and floods.

Weights are applied only in training and are normalised to mean 1.
Evaluation metrics are always computed without weights.
"""

from __future__ import annotations

import numpy as np

from llobregat.metrics import find_runs


def _normalise(w: np.ndarray) -> np.ndarray:
    w = np.where(np.isfinite(w), w, 0.0)
    m = w.mean()
    return w / m if m > 0 else np.ones_like(w)


def uniform(q) -> np.ndarray:
    return np.ones(len(np.asarray(q)), dtype=float)


def power(q, alpha: float, q_ref: float | None = None) -> np.ndarray:
    """w = (Q / Q_ref)^alpha. alpha = 0 equival a pes uniforme."""
    q = np.clip(np.asarray(q, dtype=float), 0, None)
    ref = q_ref if q_ref is not None else np.nanmean(q)
    return _normalise((q / ref) ** alpha)


def threshold(q, thr: float, c: float) -> np.ndarray:
    """w = 1 + (c − 1)·1[Q > threshold]."""
    q = np.asarray(q, dtype=float)
    return _normalise(1 + (c - 1) * (q > thr))


def event_window(q, thr: float, c: float, before_steps: int, after_steps: int = 0) -> np.ndarray:
    """Weight c in the hours around each episode, including the preceding rise."""
    q = np.asarray(q, dtype=float)
    mask = np.zeros(len(q), dtype=bool)
    for s, e in find_runs(np.nan_to_num(q, nan=-np.inf) > thr):
        mask[max(0, s - before_steps) : min(len(q), e + after_steps + 1)] = True
    return _normalise(1 + (c - 1) * mask)


def tfg_stepped(
    q,
    weights_by_quantile: dict[float, float] | None = None,
    surge_weight: float = 100.0,
    surge_min_q: float = 5.0,
    surge_quantile: float = 0.95,
) -> np.ndarray:
    """Bachelor thesis (TFG) scheme: stepped weight by target quantiles (scripts 17/18).

    By default 3/5/15/40 above Q75/Q90/Q95/Q99, and 100 if Q > 5 m³/s and Q > Q95.
    Kept to compare the new system with the thesis system on equal terms.
    Here it is normalised to mean 1 (the optimum does not change, the loss scale does).
    """
    q = np.asarray(q, dtype=float)
    steps = weights_by_quantile or {0.75: 3.0, 0.90: 5.0, 0.95: 15.0, 0.99: 40.0}
    w = np.ones(len(q))
    for tau in sorted(steps):
        w[q > np.nanquantile(q, tau)] = steps[tau]
    w[(q > surge_min_q) & (q > np.nanquantile(q, surge_quantile))] = surge_weight
    return _normalise(w)


SCHEMES = {
    "uniform": uniform,
    "power": power,
    "threshold": threshold,
    "event_window": event_window,
    "tfg_stepped": tfg_stepped,
}

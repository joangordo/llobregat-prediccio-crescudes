"""Continuous, event and interval metrics, and statistical comparison."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd


def _clean(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.all([np.isfinite(a) for a in arrs], axis=0)
    return [a[mask] for a in arrs]


# --------------------------------------------------------------------------- #
# Continuous
# --------------------------------------------------------------------------- #
def rmse(obs, pred) -> float:
    o, p = _clean(obs, pred)
    return float(np.sqrt(np.mean((o - p) ** 2)))


def mae(obs, pred) -> float:
    o, p = _clean(obs, pred)
    return float(np.mean(np.abs(o - p)))


def nse(obs, pred) -> float:
    o, p = _clean(obs, pred)
    return float(1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))


def nse_log(obs, pred, eps: float = 0.1) -> float:
    o, p = _clean(obs, pred)
    return nse(np.log(o + eps), np.log(np.clip(p, 0, None) + eps))


def kge(obs, pred) -> dict[str, float]:
    """Kling-Gupta (Gupta et al., 2009) with its components r, alpha, beta."""
    o, p = _clean(obs, pred)
    r = float(np.corrcoef(o, p)[0, 1])
    alpha = float(p.std() / o.std())
    beta = float(p.mean() / o.mean())
    value = 1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"kge": value, "r": r, "alpha": alpha, "beta": beta}


def persistence_index(obs, pred, persistence) -> float:
    """PI = 1 − SSE(model)/SSE(persistence). PI > 0: the model beats persistence."""
    o, p, ps = _clean(obs, pred, persistence)
    return float(1 - np.sum((o - p) ** 2) / np.sum((o - ps) ** 2))


def continuous_scores(obs, pred, persistence=None) -> dict[str, float]:
    out = {
        "rmse": rmse(obs, pred),
        "mae": mae(obs, pred),
        "nse": nse(obs, pred),
        "nse_log": nse_log(obs, pred),
    }
    out.update(kge(obs, pred))
    if persistence is not None:
        out["pi"] = persistence_index(obs, pred, persistence)
    return out


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def find_runs(mask) -> list[tuple[int, int]]:
    """Contiguous runs where mask is true, as inclusive (start, end) pairs."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def event_scores(obs, alert, threshold: float, lead_steps: int = 0) -> dict[str, float]:
    """POD, FAR and CSI by episode.

    Episode: continuous run with obs > threshold. Hit: the warning is on at some point between
    `lead_steps` before the start of the episode and its peak. False alarm: a warning run that
    touches no episode (each episode extended by lead_steps at the front).
    """
    o = np.nan_to_num(np.asarray(obs, dtype=float), nan=-np.inf)
    a = np.asarray(alert, dtype=bool)
    episodes = find_runs(o > threshold)
    hits = 0
    windows = []
    for s, e in episodes:
        peak = s + int(np.argmax(o[s : e + 1]))
        lo = max(0, s - lead_steps)
        if a[lo : peak + 1].any():
            hits += 1
        windows.append((lo, e))
    misses = len(episodes) - hits
    false_alarms = sum(
        1 for s, e in find_runs(a) if not any(s <= we and e >= ws for ws, we in windows)
    )
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else np.nan
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "pod": pod,
        "far": far,
        "csi": csi,
    }


def episode_table(
    obs: pd.Series,
    pred: pd.Series,
    threshold: float,
    search_steps: int = 12,
    alert: pd.Series | None = None,
) -> pd.DataFrame:
    """Table per episode: observed and forecast peak, relative error and peak timing error."""
    o = obs.to_numpy(dtype=float)
    p = pred.reindex(obs.index).to_numpy(dtype=float)
    step_h = (obs.index[1] - obs.index[0]).total_seconds() / 3600 if len(obs) > 1 else np.nan
    rows = []
    for s, e in find_runs(np.nan_to_num(o, nan=-np.inf) > threshold):
        peak = s + int(np.nanargmax(o[s : e + 1]))
        lo, hi = max(0, s - search_steps), min(len(o) - 1, e + search_steps)
        seg = p[lo : hi + 1]
        if np.all(np.isnan(seg)):
            p_peak, p_idx = np.nan, None
        else:
            p_idx = lo + int(np.nanargmax(seg))
            p_peak = p[p_idx]
        rows.append(
            {
                "start": obs.index[s],
                "end": obs.index[e],
                "peak_time": obs.index[peak],
                "peak_obs": o[peak],
                "peak_pred": p_peak,
                "peak_rel_error": (p_peak - o[peak]) / o[peak] if o[peak] else np.nan,
                "peak_timing_error_h": (p_idx - peak) * step_h if p_idx is not None else np.nan,
                "alert_in_time": bool(alert.iloc[max(0, s - search_steps) : peak + 1].any())
                if alert is not None
                else None,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Intervals and quantiles
# --------------------------------------------------------------------------- #
def coverage(obs, lo, hi) -> float:
    o, lo_, hi_ = _clean(obs, lo, hi)
    return float(np.mean((o >= lo_) & (o <= hi_)))


def mean_width(lo, hi) -> float:
    lo_, hi_ = _clean(lo, hi)
    return float(np.mean(hi_ - lo_))


def interval_score(obs, lo, hi, alpha: float) -> float:
    """Interval score / Winkler (Gneiting and Raftery, 2007). Lower is better."""
    o, lo_, hi_ = _clean(obs, lo, hi)
    s = (hi_ - lo_) + (2 / alpha) * (lo_ - o) * (o < lo_) + (2 / alpha) * (o - hi_) * (o > hi_)
    return float(np.mean(s))


def pinball(obs, q, tau: float) -> float:
    o, qq = _clean(obs, q)
    d = o - qq
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


# --------------------------------------------------------------------------- #
# Statistical comparison
# --------------------------------------------------------------------------- #
def block_bootstrap_diff(
    metric: Callable[[np.ndarray, np.ndarray], float],
    obs,
    pred_a,
    pred_b,
    groups,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """CI of metric(A) − metric(B), resampling whole groups (hydrological years)."""
    obs, pred_a, pred_b, groups = map(np.asarray, (obs, pred_a, pred_b, groups))
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in sample])
        diffs[b] = metric(obs[idx], pred_a[idx]) - metric(obs[idx], pred_b[idx])
    a = (1 - level) / 2
    return {
        "diff": float(metric(obs, pred_a) - metric(obs, pred_b)),
        "ci_low": float(np.quantile(diffs, a)),
        "ci_high": float(np.quantile(diffs, 1 - a)),
    }


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict[str, float]:
    """Diebold-Mariano test with Newey-West variance (h−1 lags)."""
    la, lb = _clean(loss_a, loss_b)
    d = la - lb
    n = len(d)
    d_c = d - d.mean()
    var = np.dot(d_c, d_c) / n
    for k in range(1, h):
        w = 1 - k / h
        var += 2 * w * np.dot(d_c[k:], d_c[:-k]) / n
    stat = d.mean() / math.sqrt(var / n)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(stat) / math.sqrt(2))))
    return {"dm_stat": float(stat), "p_value": float(p), "mean_loss_diff": float(d.mean())}

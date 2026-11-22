"""Feature engineering and target construction.

Ground rule: the row for time t may only use data up to t.
Every operation is causal (shifts into the past, right-aligned rolling windows
and forward recursions). tests/test_features.py checks it.
The same function builds the features for training and in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from llobregat.config import StationConfig


@dataclass
class FeatureSpec:
    target_col: str
    resolution_min: int = 30
    own_lags_h: list[float] = field(default_factory=lambda: [0.5, 1, 1.5, 2, 3, 6, 12, 24])
    own_windows_h: list[float] = field(default_factory=lambda: [3, 6, 24])
    upstream: dict[str, tuple[str, list[float]]] = field(
        default_factory=dict
    )  # name -> (column, lags in h)
    reservoir_outflow: dict[str, str] = field(default_factory=dict)
    reservoir_volume: dict[str, str] = field(default_factory=dict)
    rain_cols: dict[str, str] = field(default_factory=dict)  # gauge code -> column
    subbasins: dict[str, list[str]] = field(default_factory=dict)
    rain_windows_h: list[float] = field(default_factory=lambda: [0.5, 1, 3, 6, 12, 24, 72])
    per_station_rain: bool = True
    per_subbasin_rain: bool = True
    antecedent_kd: list[float] = field(default_factory=lambda: [0.85, 0.90, 0.95])
    significant_rain_mm: float = 0.5

    def steps(self, hours: float) -> int:
        return round(hours * 60 / self.resolution_min)

    @property
    def steps_per_day(self) -> int:
        return round(24 * 60 / self.resolution_min)

    @classmethod
    def from_config(
        cls,
        cfg: StationConfig,
        target_col: str,
        upstream_cols: dict[str, str],
        rain_cols: dict[str, str],
        reservoir_outflow: dict[str, str] | None = None,
        reservoir_volume: dict[str, str] | None = None,
        default_upstream_lags_h: list[float] | None = None,
    ) -> FeatureSpec:
        """Build the specification from the YAML and the actual column names."""
        # Up to 24 h: the peak-to-peak travel time from Guardiola, Súria and Balsareny to
        # Sant Joan Despí is 10-15 h (p75 up to ~20 h; llobregat analyse-features)
        default_lags = default_upstream_lags_h or [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 24]
        upstream = {}
        for gauge in cfg.upstream:
            if gauge.name not in upstream_cols:
                continue
            if gauge.lag_window_h is not None:
                lo, hi = gauge.lag_window_h
                n = cfg.steps(hi) - cfg.steps(lo)
                lags = [lo + i * cfg.resolution_min / 60 for i in range(n + 1)]
            else:
                lags = default_lags
            upstream[gauge.name] = (upstream_cols[gauge.name], lags)
        return cls(
            target_col=target_col,
            resolution_min=cfg.resolution_min,
            upstream=upstream,
            reservoir_outflow=reservoir_outflow or {},
            reservoir_volume=reservoir_volume or {},
            rain_cols={
                k: v
                for k, v in rain_cols.items()
                if k in cfg.rain.xema_stations or k in cfg.rain.aca_gauges
            },
            subbasins=cfg.rain.subbasins,
            rain_windows_h=cfg.rain.windows_h,
            per_station_rain=cfg.rain.per_station_features,
            per_subbasin_rain=cfg.rain.per_subbasin_features,
            antecedent_kd=cfg.antecedent_kd,
        )


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #
def _hlabel(h: float) -> str:
    return f"{h:g}h".replace(".", "p")


def antecedent_index(precip: pd.Series, k: float) -> pd.Series:
    """Antecedent precipitation index: API_t = k·API_{t-1} + P_t (NaN treated as 0)."""
    p = precip.fillna(0.0).to_numpy(dtype=float)
    out = np.empty_like(p)
    acc = 0.0
    for i, v in enumerate(p):
        acc = k * acc + v
        out[i] = acc
    return pd.Series(out, index=precip.index)


def hours_since(mask: pd.Series, resolution_min: int) -> pd.Series:
    """Hours since the last time `mask` was true (NaN if it has not happened yet)."""
    idx = np.arange(len(mask), dtype=float)
    last = pd.Series(np.where(mask.to_numpy(), idx, np.nan), index=mask.index).ffill()
    return (pd.Series(idx, index=mask.index) - last) * resolution_min / 60


def own_flow_features(q: pd.Series, spec: FeatureSpec) -> dict[str, pd.Series]:
    f = {"q_now": q}
    for h in spec.own_lags_h:
        f[f"q_lag_{_hlabel(h)}"] = q.shift(spec.steps(h))
    f["q_diff_1h"] = q - q.shift(spec.steps(1))
    f["q_diff_3h"] = q - q.shift(spec.steps(3))
    for h in spec.own_windows_h:
        n = spec.steps(h)
        f[f"q_mean_{_hlabel(h)}"] = q.rolling(n, min_periods=1).mean()
        f[f"q_max_{_hlabel(h)}"] = q.rolling(n, min_periods=1).max()
    return f


def upstream_features(wide: pd.DataFrame, spec: FeatureSpec) -> dict[str, pd.Series]:
    f = {}
    for name, (col, lags) in spec.upstream.items():
        s = wide[col]
        for h in lags:
            f[f"up_{name}_lag_{_hlabel(h)}"] = s.shift(spec.steps(h))
        f[f"up_{name}_diff_1h"] = s - s.shift(spec.steps(1))
    return f


def reservoir_features(wide: pd.DataFrame, spec: FeatureSpec) -> dict[str, pd.Series]:
    f = {}
    for name, col in spec.reservoir_outflow.items():
        s = wide[col]
        f[f"res_{name}_out_now"] = s
        f[f"res_{name}_out_mean_6h"] = s.rolling(spec.steps(6), min_periods=1).mean()
        f[f"res_{name}_out_diff_3h"] = s - s.shift(spec.steps(3))
    for name, col in spec.reservoir_volume.items():
        s = wide[col]
        f[f"res_{name}_vol_now"] = s
        f[f"res_{name}_vol_diff_6h"] = s - s.shift(spec.steps(6))
        f[f"res_{name}_vol_diff_24h"] = s - s.shift(spec.steps(24))
    return f


def rain_features(wide: pd.DataFrame, spec: FeatureSpec) -> dict[str, pd.Series]:
    f: dict[str, pd.Series] = {}
    if not spec.rain_cols:
        return f
    rain = pd.DataFrame({code: wide[col] for code, col in spec.rain_cols.items()})

    def accumulations(s: pd.Series, prefix: str) -> None:
        for h in spec.rain_windows_h:
            f[f"{prefix}_sum_{_hlabel(h)}"] = s.rolling(spec.steps(h), min_periods=1).sum()

    if spec.per_station_rain:
        for code in rain.columns:
            accumulations(rain[code], f"rain_st_{code}")

    if spec.per_subbasin_rain:
        for name, codes in spec.subbasins.items():
            present = [c for c in codes if c in rain.columns]
            if not present:
                continue
            mean = rain[present].mean(axis=1, skipna=True)
            accumulations(mean, f"rain_sb_{name}")
            f[f"rain_sb_{name}_imax_6h"] = mean.rolling(spec.steps(6), min_periods=1).max()

    basin = rain.mean(axis=1, skipna=True)
    accumulations(basin, "rain_basin")
    for kd in spec.antecedent_kd:
        k = kd ** (1 / spec.steps_per_day)
        f[f"api_kd{kd:g}".replace(".", "p")] = antecedent_index(basin, k)
    f["hours_since_rain"] = hours_since(basin >= spec.significant_rain_mm, spec.resolution_min)
    return f


def calendar_features(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    doy = index.day_of_year.to_numpy()
    angle = 2 * np.pi * (doy - 1) / 365.25
    return {
        "doy_sin": pd.Series(np.sin(angle), index=index),
        "doy_cos": pd.Series(np.cos(angle), index=index),
    }


def build_features(wide: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Feature table (one row per time t, using only information up to t)."""
    q = wide[spec.target_col]
    blocks: dict[str, pd.Series] = {}
    blocks.update(own_flow_features(q, spec))
    blocks.update(upstream_features(wide, spec))
    blocks.update(reservoir_features(wide, spec))
    blocks.update(rain_features(wide, spec))
    blocks.update(calendar_features(wide.index))
    return pd.DataFrame(blocks, index=wide.index)


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
def make_target(q: pd.Series, h_steps: int, transform: str, eps: float = 0.1) -> pd.Series:
    """Target for horizon h: value at t that depends on Q(t+h) and Q(t)."""
    future = q.shift(-h_steps)
    if transform == "level":
        return future
    if transform == "delta":
        return future - q
    if transform == "log_ratio":
        return np.log(future + eps) - np.log(q + eps)
    if transform == "log_level":
        return np.log(future + eps)
    raise ValueError(f"Unknown transform: {transform}")


def forward_target(q_future, q_now, transform: str, eps: float = 0.1) -> np.ndarray:
    """Vector version of make_target: transformed target from Q(t+h) and Q(t)."""
    f = np.asarray(q_future, dtype=float)
    q = np.asarray(q_now, dtype=float)
    if transform == "level":
        return f
    if transform == "delta":
        return f - q
    if transform == "log_ratio":
        return np.log(f + eps) - np.log(q + eps)
    if transform == "log_level":
        return np.log(f + eps)
    raise ValueError(f"Unknown transform: {transform}")


def inverse_target(y, q_now, transform: str, eps: float = 0.1):
    """Undo the transform (also valid for quantiles: all of them are monotonic)."""
    y = np.asarray(y, dtype=float)
    q_now = np.asarray(q_now, dtype=float)
    if transform == "level":
        q = y
    elif transform == "delta":
        q = q_now + y
    elif transform == "log_ratio":
        q = np.exp(y + np.log(q_now + eps)) - eps
    elif transform == "log_level":
        q = np.exp(y) - eps
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return np.clip(q, 0.0, None)


# --------------------------------------------------------------------------- #
# Travel time between stations
# --------------------------------------------------------------------------- #
def estimate_travel_time(
    upstream: pd.Series,
    downstream: pd.Series,
    max_lag_steps: int,
    use_diff: bool = True,
    min_quantile: float | None = None,
) -> tuple[int, pd.Series]:
    """Lag (in steps) that maximises the correlation between an upstream and a downstream gauge.

    use_diff: correlate the differences (removes the trend and highlights the rises).
    min_quantile: if given, only use time steps with downstream flow above this quantile
    (to estimate the travel time during floods, which is faster).
    """
    up = upstream.diff() if use_diff else upstream
    down = downstream.diff() if use_diff else downstream
    mask = pd.Series(True, index=downstream.index)
    if min_quantile is not None:
        mask = downstream >= downstream.quantile(min_quantile)
    corr = {}
    for lag in range(max_lag_steps + 1):
        pair = pd.concat([up.shift(lag), down], axis=1)[mask].dropna()
        corr[lag] = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) > 10 else np.nan
    corr_s = pd.Series(corr, name="corr")
    return int(corr_s.idxmax()), corr_s


def lag_window(
    tau_min_h: float, tau_max_h: float, horizon_h: float, margin_h: float = 2.0
) -> tuple[float, float]:
    """Lag window (h) for a horizon: [max(0, τmin−h−m), max(0, τmax−h+m)]."""
    return max(0.0, tau_min_h - horizon_h - margin_h), max(0.0, tau_max_h - horizon_h + margin_h)

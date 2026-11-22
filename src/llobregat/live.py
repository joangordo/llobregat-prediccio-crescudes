"""Live forecast: from the ACA real-time API (Sentilo) to forecasts with the operational model.

The training data were built from the ACA historical archive; in real time the same inputs
come from Sentilo at 5-minute resolution. This module turns those readings into exactly the
same 30-minute table the model was trained on:

  flows       quality control with the training rules (and the training quantiles), 30-min
              mean; upstream gaps of up to 3 h are interpolated, the target is never filled
  reservoirs  mean of the previous complete day (the training used the official daily
              series, which in real time is only published the next day)
  rain        5-minute intensity (mm/h) / 12 = mm, label shifted +5 min to the end of the
              interval, summed to 30 min (as with the historical rain gauge files)

then builds the features with llobregat.features and calls OperationalModel.predict.

A rolling cache of raw readings (CACHE_DAYS) keeps each run incremental: the antecedent
precipitation index needs a few weeks of rain, the other features at most 72 h.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import StationConfig
from llobregat.features import FeatureSpec, build_features
from llobregat.qc import fill_short_gaps
from llobregat.resample import to_grid

log = logging.getLogger("live")

CACHE_DAYS = 60
FREQ = "30min"
MAX_RAIN_5MIN_MM = 30.0
MAX_RAIN_30MIN_MM = 100.0
MIN_DAILY_COVERAGE = 0.5  # share of a day's 5-min readings needed for a daily mean


@dataclass(frozen=True)
class LiveSeries:
    key: str  # column of the 30-min table: q, up:<name>, res:<name>:out|vol, rain:<code>
    kind: str  # flow | outflow | volume | rain
    provider: str
    sensor: str


def _ok(ref) -> bool:
    return ref is not None and "TODO" not in (ref.provider, ref.sensor)


def live_series(cfg: StationConfig, inputs: dict[str, list[str]]) -> list[LiveSeries]:
    """Real-time series needed by a model trained with `inputs` (manifest["inputs"])."""
    out = [LiveSeries("q", "flow", cfg.target.provider, cfg.target.sensor)]
    gauges = {g.name: g for g in cfg.upstream}
    for name in inputs.get("flows", []):
        g = gauges[name].aca
        out.append(LiveSeries(f"up:{name}", "flow", g.provider, g.sensor))
    reservoirs = {r.name: r for r in cfg.reservoirs}
    for kind, key, attr in (
        ("reservoir_outflow", "out", "outflow"),
        ("reservoir_volume", "vol", "volume"),
    ):
        for name in inputs.get(kind, []):
            ref = getattr(reservoirs[name], attr)
            out.append(LiveSeries(f"res:{name}:{key}", attr, ref.provider, ref.sensor))
    missing = []
    for code in inputs.get("rain_aca", []):
        ref = cfg.rain.aca_realtime.get(code)
        if _ok(ref):
            out.append(LiveSeries(f"rain:{code}", "rain", ref.provider, ref.sensor))
        else:
            missing.append(code)
    if missing:
        log.warning(
            "No real-time sensor configured for rain gauges %s (rain.aca_realtime)", missing
        )
    if inputs.get("rain_xema"):
        log.warning("XEMA rain gauges are not available in real time without a Meteocat key")
    return out


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


# --------------------------------------------------------------------------- #
# Rolling cache of raw readings
# --------------------------------------------------------------------------- #
def update_cache(
    series: list[LiveSeries],
    cache_dir: Path,
    client,
    now: datetime | None = None,
    days: int = CACHE_DAYS,
) -> dict[str, pd.DataFrame]:
    """Fetch the readings after the last cached one (or the last `days` days) and trim.

    Returns {key: DataFrame(timestamp, value)}. A series that fails keeps its cache.
    """
    from llobregat.ingest.aca_sentilo import observations_to_frame

    now = now or datetime.now(UTC)
    start_all = pd.Timestamp(now) - pd.Timedelta(days=days)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in series:
        f = cache_dir / f"{_safe(s.key)}.parquet"
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["timestamp", "value"])
        start = start_all
        if len(old):
            start = max(start_all, pd.Timestamp(old["timestamp"].max()) + pd.Timedelta(seconds=1))
        try:
            obs = client.observations_range(
                s.provider, s.sensor, start=start.to_pydatetime(), end=now
            )
            new = observations_to_frame(obs, s.provider, s.sensor, s.kind)[["timestamp", "value"]]
        except RuntimeError as exc:
            log.error("%s (%s/%s): %s", s.key, s.provider, s.sensor, exc)
            new = old.iloc[:0]
        parts = [p for p in (old, new) if len(p)]
        df = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC"), "value": []})
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = (
            df[df["timestamp"] >= start_all]
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df.to_parquet(f, index=False)
        out[s.key] = df
        log.info("%s: %d new readings, %d cached", s.key, len(new), len(df))
    return out


# --------------------------------------------------------------------------- #
# 30-minute table
# --------------------------------------------------------------------------- #
def _flow(df: pd.DataFrame, cfg: StationConfig, ref: dict | None) -> pd.Series:
    from llobregat.qc_flow import apply_flow_qc

    if df.empty:
        return pd.Series(dtype=float)
    clean, _ = apply_flow_qc(df[["timestamp", "value"]], None, cfg.qc.flow, reference=ref)
    s = clean.set_index("timestamp")["value_qc"].dropna()
    return to_grid(s, "mean", FREQ) if len(s) else pd.Series(dtype=float)


def _previous_day(df: pd.DataFrame, grid: pd.DatetimeIndex, step_min: float = 5) -> pd.Series:
    from llobregat.dataset import previous_day

    if df.empty:
        return pd.Series(np.nan, index=grid)
    s = df.set_index("timestamp")["value"].astype(float)
    daily = s.resample("D").mean()
    n = s.resample("D").count()
    daily = daily.where(n >= MIN_DAILY_COVERAGE * 24 * 60 / step_min)
    return previous_day(daily.dropna(), grid)


def _rain(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("timestamp")["value"].astype(float) / 12
    s.index = s.index + pd.Timedelta(minutes=5)
    s = s[(s >= 0) & (s <= MAX_RAIN_5MIN_MM)]
    out = to_grid(s, "sum", FREQ)
    return out.where(out <= MAX_RAIN_30MIN_MM)


def live_table(
    cfg: StationConfig,
    series: list[LiveSeries],
    raw: dict[str, pd.DataFrame],
    now: datetime | None = None,
    days: int = CACHE_DAYS,
    ranges: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """The 30-min table of raw inputs (same columns as in training), up to `now`."""
    now = pd.Timestamp(now or datetime.now(UTC))
    end = now.floor(FREQ)
    grid = pd.date_range(end - pd.Timedelta(days=days), end, freq=FREQ)
    ranges = ranges or {}
    cols = {}
    for s in series:
        df = raw.get(s.key, pd.DataFrame(columns=["timestamp", "value"]))
        if s.kind == "flow":
            g = _flow(df, cfg, ranges.get(s.key)).reindex(grid)
            if s.key != "q":
                g = fill_short_gaps(g, cfg.qc.short_gap_intervals)
        elif s.kind in ("outflow", "volume"):
            g = _previous_day(df, grid)
        else:
            g = _rain(df).reindex(grid)
        cols[s.key] = g.astype(float)
    return pd.DataFrame(cols, index=grid)


def live_features(cfg: StationConfig, manifest: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Features of every row of the live table, in the order the model expects."""
    inputs = manifest.get("inputs", {})
    spec = FeatureSpec.from_config(
        cfg,
        target_col="q",
        upstream_cols={n: f"up:{n}" for n in inputs.get("flows", [])},
        rain_cols={
            c: f"rain:{c}" for c in inputs.get("rain_aca", []) + inputs.get("rain_xema", [])
        },
        reservoir_outflow={n: f"res:{n}:out" for n in inputs.get("reservoir_outflow", [])},
        reservoir_volume={n: f"res:{n}:vol" for n in inputs.get("reservoir_volume", [])},
    )
    needed = [
        *(col for col, _ in spec.upstream.values()),
        *spec.rain_cols.values(),
        *spec.reservoir_outflow.values(),
        *spec.reservoir_volume.values(),
    ]
    table = table.copy()
    for col in needed:
        if col not in table:
            table[col] = np.nan
    feats = build_features(table, spec)
    if "hours_since_rain" in feats:
        # In training the history went back years; here a long dry spell may not contain any
        # rain at all: use the length of the cache as a lower bound
        span_h = (feats.index - feats.index[0]).total_seconds() / 3600
        feats["hours_since_rain"] = feats["hours_since_rain"].fillna(
            pd.Series(span_h, index=feats.index)
        )
    for col in manifest["features"]:
        if col not in feats:
            feats[col] = np.nan
    return feats[manifest["features"]]


def input_status(
    series: list[LiveSeries],
    raw: dict[str, pd.DataFrame],
    table: pd.DataFrame,
    ranges: dict[str, dict],
    now: datetime | None = None,
) -> pd.DataFrame:
    """Freshness and plausibility of each input, for the app."""
    now = pd.Timestamp(now or datetime.now(UTC))
    rows = []
    for s in series:
        df = raw.get(s.key, pd.DataFrame(columns=["timestamp", "value"]))
        last = pd.Timestamp(df["timestamp"].max()) if len(df) else None
        value = table[s.key].dropna().iloc[-1] if table[s.key].notna().any() else np.nan
        r = ranges.get(s.key, {})
        check = "ok"
        if last is None:
            check = "no data"
        elif r and np.isfinite(value) and r.get("max", 0) > 0 and value > 20 * r["max"]:
            check = "out of range (units?)"
        rows.append(
            {
                "key": s.key,
                "kind": s.kind,
                "sensor": f"{s.provider}/{s.sensor}",
                "last_reading": None if last is None else last.isoformat(),
                "age_min": None if last is None else round((now - last).total_seconds() / 60),
                "value": None if not np.isfinite(value) else round(float(value), 3),
                "check": check,
            }
        )
    return pd.DataFrame(rows)


def forecast(
    cfg: StationConfig,
    model,
    raw: dict[str, pd.DataFrame],
    series: list[LiveSeries],
    now: datetime | None = None,
    history_h: float = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts issued over the last `history_h` hours (rows with an observed current flow).

    Returns (forecasts, table): the forecasts of OperationalModel.predict and the 30-min table
    of inputs they were made from.
    """
    ranges = model.manifest.get("input_ranges", {})
    table = live_table(cfg, series, raw, now=now, ranges=ranges)
    X = live_features(cfg, model.manifest, table)
    recent = X[X.index >= X.index[-1] - pd.Timedelta(hours=history_h)]
    recent = recent[recent["q_now"].notna()]
    if recent.empty:
        return pd.DataFrame(), table
    return model.predict(recent), table

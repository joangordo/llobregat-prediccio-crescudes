"""Quality control of the 5-min flow series (after chaining and unit correction).

The rules were defined from real cases reviewed with qc-event. Each rule flags readings;
the clean value (value_qc) is NaN wherever there is a flag, and the original is kept.

  spike          a value > 4 x the median of the 3 before and the 3 after (and > the
                 station p99): Balsareny 2008-10-17, Cardona 2007-02-07, Aigua de Valls
                 2026-10-28
  dropout        a value < 1/10 of its neighbours while the river carries water (isolated
                 zero readings) or a run of zeros that recovers within 6 h (Guardiola
                 1997-12-18)
  excursion      a rise of more than x10 that returns to the previous level within 3 h:
                 Balsareny 1997-10-30, Cardona 2007-02-04, Merlès 2017-09-27, Mediona 2026-10-03
  oscillation    3 or more x20 jumps within an hour (two levels alternating):
                 Mediona 2026-10-03, riera de Carme 2009-12-14
  stuck          the same value for 7 days or more with > 0.5 m³/s (stuck sensor):
                 Castellbisbal. A dry river showing the same minimum flow is not flagged.
  high_flat      the same value for 6 h or more with flow above p95 (saturated sensor):
                 Monegal 2019-12-20
  unit_step      a x100 jump exactly where the unit factor changes: both days are flagged
  daily_balance  the daily mean (with ≥80 % of the day covered) does not match the official
                 daily flow (x3 or /3, trying UTC and local days): shifted, stuck or
                 wrong-unit stretches (Sant Joan Despí 1999, sensor stuck at 140). Systematic
                 x2 differences (rating curve revised in the daily series) are not flagged.
  exclusion      stretches discarded by hand in the configuration (qc.exclusions)

The daily balance needs the official daily flow, which does not exist in real time until
the next day: in operation the other rules are applied.

  uv run llobregat qc-apply [--config ...]

Output in data/interim/aca_flow_qc/: <code>.parquet (value_qc, flags), summary.csv,
balance_days.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import FlowQCConfig, StationConfig, load_station_config
from llobregat.ingest.aca_merge import configured_series
from llobregat.ingest.aca_opendata import safe_name

log = logging.getLogger("qc_flow")

RULES = [
    "excursion",
    "spike",
    "dropout",
    "oscillation",
    "stuck",
    "high_flat",
    "unit_step",
    "daily_balance",
    "exclusion",
]


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def neighbour_medians(x: pd.Series, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    before = x.shift(1).rolling(k, min_periods=1).median().to_numpy()
    after = x[::-1].shift(1).rolling(k, min_periods=1).median()[::-1].to_numpy()
    return before, after


def flag_spikes(x: pd.Series, p: FlowQCConfig, p99: float) -> tuple[np.ndarray, np.ndarray]:
    before, after = neighbour_medians(x)
    v = x.to_numpy()
    hi_ref = np.fmax(before, after)
    lo_ref = np.fmin(before, after)
    spike = (v > p.spike_factor * hi_ref + p.spike_min_m3s) & (v > p99)
    drop = (lo_ref > p.dropout_min_m3s) & (v < lo_ref / p.dropout_factor)
    return np.nan_to_num(spike, nan=False).astype(bool), np.nan_to_num(drop).astype(bool)


def _window_mark(t: pd.DatetimeIndex, end_mark: np.ndarray, width: str) -> np.ndarray:
    """Flag the readings that fall within any flagged window (t_end - width, t_end]."""
    if not end_mark.any():
        return np.zeros(len(t), dtype=bool)
    ns = t.as_unit("ns").asi8  # pandas 3 may store it in µs
    rev = pd.Series(
        end_mark[::-1].astype(float), index=pd.to_datetime(ns[-1] - ns[::-1], unit="ns")
    )
    hit = rev.rolling(width, closed="both").max().to_numpy()[::-1]
    return hit > 0


def flag_oscillation(t: pd.DatetimeIndex, x: pd.Series, p: FlowQCConfig) -> np.ndarray:
    """3 or more x20 jumps between consecutive readings within any hour."""
    v = x.to_numpy()
    prev = np.r_[np.nan, v[:-1]]
    lo = np.fmax(np.fmin(v, prev), 1e-3)
    jump = ((np.fmax(v, prev) / lo) > p.oscillation_ratio).astype(float)
    n = pd.Series(jump, index=t).rolling("1h").sum().to_numpy()
    return _window_mark(t, n >= p.oscillation_min_jumps, "1h")


def flag_zero_blocks(t: pd.DatetimeIndex, x: pd.Series, p: FlowQCConfig) -> np.ndarray:
    """Run of zeros (≤ 0.02 m³/s) while the river carries water, recovering within 6 h: the
    sensor stopped measuring (Guardiola 1997-12-18, in the middle of a flood)."""
    v = x.to_numpy()
    pre = pd.Series(v, index=t).rolling("1h", closed="left").median().to_numpy()
    prev = np.r_[np.nan, v[:-1]]
    start = (v <= p.zero_block_max_m3s) & (prev > p.zero_block_max_m3s)  # start of the zero run
    cand = np.flatnonzero(start & (pre > p.dropout_min_m3s))
    out = np.zeros(len(v), dtype=bool)
    ns = t.as_unit("ns").asi8
    hour = int(pd.Timedelta(hours=1).value)
    horizon = int(pd.Timedelta(hours=p.zero_block_return_hours).value)
    for i in cand:
        if out[i]:
            continue
        # The level before must be real water: at least 3 distinct values above zero in the
        # previous hour (a fixed block of 230.00 with interleaved zeros is not)
        j0 = np.searchsorted(ns, ns[i] - hour, side="left")
        before = v[j0:i]
        if len(np.unique(before[before > p.zero_block_max_m3s])) < 3:
            continue
        k = np.searchsorted(ns, ns[i] + horizon, side="right")
        back = np.flatnonzero(v[i + 1 : k] >= pre[i] / 2)
        if len(back):
            out[i : i + 1 + back[0]] = True
    return out


def flag_excursions(t: pd.DatetimeIndex, x: pd.Series, p: FlowQCConfig, p99: float) -> np.ndarray:
    """Rise of more than x10 over the level of the previous hour that returns to less than
    twice that level within 3 h: a real river does not fall that fast (Balsareny 1997,
    Cardona 2007, Merlès 2017, Mediona 2026)."""
    v = x.to_numpy()
    pre_s = pd.Series(v, index=t).rolling("1h", closed="left").median()
    # If the previous hour has no readings (gap), take the last known level
    last = pd.Series(v, index=t).shift(1).ffill()
    pre = pre_s.fillna(last).to_numpy()
    cand = np.flatnonzero((v > p.excursion_factor * np.fmax(pre, 1e-3)) & (v > p99))
    out = np.zeros(len(v), dtype=bool)
    if not len(cand):
        return out
    ns = t.as_unit("ns").asi8  # pandas 3 may store it in µs
    horizon = int(pd.Timedelta(hours=p.excursion_return_hours).value)
    for i in cand:
        if out[i]:
            continue
        k = np.searchsorted(ns, ns[i] + horizon, side="right")
        back = np.flatnonzero(v[i + 1 : k] <= 2 * max(pre[i], 1e-3))
        if len(back):
            out[i : i + 1 + back[0]] = True
    return out


def runs(t: pd.DatetimeIndex, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identifier and duration (hours) of the run of identical values of each reading."""
    gid = np.cumsum(np.r_[True, v[1:] != v[:-1]])
    s = pd.Series(t, index=gid)
    dur = s.groupby(level=0).transform("max") - s.groupby(level=0).transform("min")
    return gid, (dur / pd.Timedelta(hours=1)).to_numpy()


def flag_flat(
    t: pd.DatetimeIndex, v: np.ndarray, p: FlowQCConfig, high: float
) -> tuple[np.ndarray, np.ndarray]:
    _, hours = runs(t, v)
    stuck = (hours >= 24 * p.stuck_days) & (v > p.stuck_min_m3s)
    high_flat = (hours >= p.high_flat_hours) & (v > high)
    return stuck, high_flat


def flag_unit_step(
    t: pd.DatetimeIndex, v: np.ndarray, factor: np.ndarray | None, p: FlowQCConfig
) -> np.ndarray:
    out = np.zeros(len(v), dtype=bool)
    if factor is None or len(v) < 2:
        return out
    change = np.r_[False, factor[1:] != factor[:-1]]
    prev = np.r_[np.nan, v[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.fmax(v, prev) / np.fmax(np.fmin(v, prev), 1e-3)
    steps = change & (ratio > p.unit_step_ratio)
    if steps.any():
        day = t.floor("D")
        bad_days = set(day[steps]) | set(day[np.r_[steps[1:], False]])
        out = np.isin(day, list(bad_days))
    return out


def flag_daily_balance(
    t: pd.DatetimeIndex, v: np.ndarray, ok: np.ndarray, daily: pd.DataFrame | None, p: FlowQCConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """The mean of the day's readings must match the official daily flow.

    It is not known whether the official day is UTC or local time: the three possible
    offsets (0, +1 and +2 h) are compared and the day is flagged only if none matches.
    """
    empty = pd.DataFrame(columns=["day", "mean_5min", "official_daily", "ratio"])
    if daily is None or daily.empty:
        return np.zeros(len(v), dtype=bool), empty
    ref_all = pd.Series(
        daily["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(daily["timestamp"]).floor("D")
    )
    ref_all = ref_all[~ref_all.index.duplicated(keep="last")]
    cap = 20 * float(np.nanquantile(v, 0.999)) if len(v) else np.inf
    ref_all = ref_all.where(ref_all <= cap)  # impossible official daily value for the station
    day0 = t.floor("D")
    bad_all, ratio0, mean0, ref0 = None, None, None, None
    for h in (0, 1, 2):
        day = (t + pd.Timedelta(hours=h)).floor("D")
        s = pd.Series(np.where(ok, v, np.nan), index=day)
        mean = s.groupby(level=0).mean()
        slots = pd.Series(t.floor("30min"), index=day).where(ok).groupby(level=0).nunique()
        ref = ref_all.reindex(mean.index)
        ratio = mean / ref
        test = (slots / 48 >= p.daily_min_coverage) & (ref > p.daily_min_m3s)
        bad = test & ((ratio > p.daily_ratio) | (ratio < 1 / p.daily_ratio))
        bad = bad.reindex(day0.unique(), fill_value=False)
        bad_all = bad if bad_all is None else (bad_all & bad)
        if h == 0:
            ratio0, mean0, ref0 = ratio, mean, ref
    bad_days = bad_all[bad_all].index
    table = pd.DataFrame(
        {
            "day": bad_days,
            "mean_5min": mean0.reindex(bad_days).round(3).to_numpy(),
            "official_daily": ref0.reindex(bad_days).round(3).to_numpy(),
            "ratio": ratio0.reindex(bad_days).round(2).to_numpy(),
        }
    )
    return np.isin(day0, bad_days), table


def apply_flow_qc(
    df: pd.DataFrame,
    daily: pd.DataFrame | None,
    p: FlowQCConfig,
    exclusions: list[tuple[pd.Timestamp, pd.Timestamp]] = (),
    reference: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag suspicious readings; value_qc is NaN where any rule fires.

    reference: station quantiles {"p99": ..., "p95": ...} to use instead of those of `df`.
    In real time the series is only a few weeks long, so its own quantiles would not be
    those of the history the rules were tuned on.
    """
    df = df.dropna(subset=["value"]).sort_values("timestamp").reset_index(drop=True)
    t = pd.DatetimeIndex(df["timestamp"])
    x = df["value"].astype(float)
    v = x.to_numpy()
    p99, p95 = float(x.quantile(0.99)), float(x.quantile(p.high_flat_quantile))
    if reference:
        p99, p95 = reference.get("p99", p99), reference.get("p95", p95)
    flags = pd.DataFrame(False, index=df.index, columns=RULES)
    flags["spike"], flags["dropout"] = flag_spikes(x, p, p99)
    flags["dropout"] |= flag_zero_blocks(t, x, p)
    # Excursions are searched without the zero runs (otherwise the return to water after a
    # zero run would look like an excursion)
    flags["excursion"] = flag_excursions(t, x.where(~flags["dropout"].to_numpy()), p, p99)
    flags["oscillation"] = flag_oscillation(t, x, p)
    flags["stuck"], flags["high_flat"] = flag_flat(t, v, p, p95)
    factor = df["unit_factor"].to_numpy() if "unit_factor" in df else None
    flags["unit_step"] = flag_unit_step(t, v, factor, p)
    for a, b in exclusions:
        flags["exclusion"] |= (t >= a) & (t <= b)
    ok = ~flags.any(axis=1).to_numpy()
    flags["daily_balance"], bal = flag_daily_balance(t, v, ok, daily, p)
    any_flag = flags.any(axis=1)
    out = df.copy()
    out["value_qc"] = np.where(any_flag, np.nan, v)
    labels = np.full(len(df), "", dtype=object)
    idx = np.flatnonzero(any_flag.to_numpy())
    if len(idx):
        sub = flags.iloc[idx]
        lab = pd.Series("", index=sub.index, dtype=object)
        for r in RULES:
            lab = lab + np.where(sub[r].to_numpy(), r + ",", "")
        labels[idx] = lab.str.rstrip(",").to_numpy()
    out["flags"] = labels
    return out, bal


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def flow_series(cfg: StationConfig) -> list[tuple[str, str]]:
    seen, out = set(), []
    for name, kind, ref in configured_series(cfg):
        if kind == "flow" and ref.opendata and ref.opendata not in seen:
            seen.add(ref.opendata)
            out.append((name, ref.opendata))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="llobregat qc-apply",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    ap.add_argument("--interim", default="data/interim")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    interim = Path(args.interim)
    out_dir = interim / "aca_flow_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, bals = [], []
    for name, code in flow_series(cfg):
        f = interim / "aca_flow" / f"{code}.parquet"
        if not f.exists():
            continue
        dfile = interim / "aca_daily" / f"{safe_name(name)}__flow.parquet"
        daily = pd.read_parquet(dfile) if dfile.exists() else None
        excl = [
            (pd.Timestamp(e.start, tz="UTC"), pd.Timestamp(e.end, tz="UTC"))
            for e in cfg.qc.exclusions
            if e.series == code
        ]
        out, bal = apply_flow_qc(pd.read_parquet(f), daily, cfg.qc.flow, excl)
        out.to_parquet(out_dir / f"{code}.parquet", index=False)
        fl = out["flags"].astype(str)
        row = {
            "name": name,
            "codi_variable": code,
            "readings": len(out),
            "flagged_%": round(100 * float((fl != "").mean()), 2),
        }
        for r in RULES:
            row[r] = int(fl.str.contains(r, regex=False).sum())
        row["balance_days"] = len(bal)
        row["max_before"] = round(float(out["value"].max()), 1)
        row["max_clean"] = round(float(np.nanmax(out["value_qc"])), 1)
        row["max_clean_time"] = f"{out.loc[out['value_qc'].idxmax(), 'timestamp']:%Y-%m-%d %H:%M}"
        rows.append(row)
        if len(bal):
            bals.append(bal.assign(name=name, codi_variable=code))
        log.info("%s: %.2f %% flagged", name, row["flagged_%"])
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    if bals:
        pd.concat(bals).to_csv(out_dir / "balance_days.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nReadings flagged per rule (a reading can have more than one):")
    print(res.drop(columns=["codi_variable"]).to_string(index=False))
    print(f"\nSaved to {out_dir}")

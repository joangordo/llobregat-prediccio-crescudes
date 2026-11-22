"""Quality-control report of the historical flows (chained 5-min series).

It does not change any data: it describes each series so that the qc thresholds can be set
with judgement.
  - range: negatives, zeros, maximum and its date, high percentiles
  - largest peaks (events at least 3 days apart), to check them against known floods
  - jumps between 30-min intervals (99.9th percentile and maximum)
  - isolated 5-min spikes (a single value far above both neighbours)
  - repeated values (flat runs) above the minimum flow
  - comparison between reviewed (NQ_REV1) and unreviewed (NQ_REV0) data

  uv run llobregat qc-report [--config configs/stations/sant_joan_despi.yaml]

Output in data/interim/qc/: report.csv, peaks.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import StationConfig, load_station_config

log = logging.getLogger("qc_report")

SPIKE_RATIO = 3.0  # a value is an isolated spike if it exceeds both neighbours 3 times...
SPIKE_MIN_M3S = 5.0  # ... and by more than 5 m³/s


def series_list(cfg: StationConfig) -> list[tuple[str, str]]:
    out = [(f"{cfg.station_id} (target)", cfg.target.opendata)]
    out += [(g.name, g.aca.opendata) for g in cfg.upstream if g.aca.opendata]
    out += [
        ((s.note or s.sensor).split(" (")[0][:40], s.opendata)
        for s in cfg.log_only
        if s.opendata and s.variable == "flow"
    ]
    return [(n, c) for n, c in out if c]


def isolated_spikes(v: np.ndarray) -> np.ndarray:
    """Values that exceed both neighbours SPIKE_RATIO times and by more than SPIKE_MIN_M3S."""
    out = np.zeros(len(v), dtype=bool)
    if len(v) < 3:
        return out
    prev, cur, nxt = v[:-2], v[1:-1], v[2:]
    neigh = np.fmax(prev, nxt)
    out[1:-1] = (cur > SPIKE_RATIO * neigh) & (cur - neigh > SPIKE_MIN_M3S)
    return out


def run_ids(v: np.ndarray) -> np.ndarray:
    """Identifier of the run of identical consecutive values of each reading."""
    return np.cumsum(np.r_[True, v[1:] != v[:-1]])


def flat_runs(s: pd.Series, min_value: float) -> pd.DataFrame:
    """Runs of identical consecutive values above min_value: start, end and duration
    (the index of the table is the run_ids identifier)."""
    v = s.to_numpy()
    gid = run_ids(v)
    g = pd.DataFrame({"t": s.index, "v": v, "g": gid})
    runs = g.groupby("g").agg(
        start=("t", "first"), end=("t", "last"), value=("v", "first"), n=("v", "size")
    )
    runs["hours"] = (runs["end"] - runs["start"]) / pd.Timedelta(hours=1)
    return runs[(runs["value"] > min_value) & (runs["n"] > 1)]


def top_events(g: pd.Series, n: int = 5, sep_days: float = 3) -> list[tuple[pd.Timestamp, float]]:
    """The n largest values of the series, at least sep_days days apart."""
    s = g.dropna().sort_values(ascending=False)
    picked: list[tuple[pd.Timestamp, float]] = []
    sep = pd.Timedelta(days=sep_days)
    for t, v in s.items():
        if all(abs(t - u) > sep for u, _ in picked):
            picked.append((t, float(v)))
            if len(picked) == n:
                break
    return picked


def describe(
    name: str, code: str, df: pd.DataFrame, flat_min: float, flat_hours: float
) -> tuple[dict, list[dict]]:
    df = df.dropna(subset=["value"]).sort_values("timestamp")
    s = df.set_index("timestamp")["value"]
    g = s.resample("30min", label="right", closed="right").mean()
    d = g.diff().abs()
    spikes = isolated_spikes(s.to_numpy())
    runs = flat_runs(s, flat_min)
    long_runs = runs[runs["hours"] >= flat_hours]
    nq = df["quality_level"].astype(str) if "quality_level" in df else pd.Series("?", df.index)
    row = {
        "name": name,
        "codi_variable": code,
        "start": s.index.min().date(),
        "end": s.index.max().date(),
        "readings": len(s),
        "negatives": int((s < 0).sum()),
        "zeros_%": round(100 * float((s == 0).mean()), 2),
        "p50": round(float(s.median()), 2),
        "p99": round(float(s.quantile(0.99)), 1),
        "p99_9": round(float(s.quantile(0.999)), 1),
        "max_5min": round(float(s.max()), 1),
        "max_time": f"{s.idxmax():%Y-%m-%d %H:%M}",
        "max_30min": round(float(g.max()), 1),
        "jump30_p99_9": round(float(d.quantile(0.999)), 1),
        "jump30_max": round(float(d.max()), 1),
        "jump_max_time": f"{d.idxmax():%Y-%m-%d %H:%M}" if d.notna().any() else "",
        "isolated_spikes": int(spikes.sum()),
        "long_flat_runs": len(long_runs),
        "longest_flat_run_h": round(float(runs["hours"].max()), 1) if len(runs) else 0.0,
    }
    # Reviewed vs unreviewed: rate of isolated spikes and of readings in long flat runs
    in_long = np.isin(run_ids(s.to_numpy()), long_runs.index.to_numpy())
    for q in ("NQ_REV0", "NQ_REV1"):
        m = (nq == q).to_numpy()
        n = int(m.sum())
        row[f"{q}_%"] = round(100 * n / len(s), 1)
        row[f"{q}_spikes_per_100k"] = round(1e5 * spikes[m].sum() / n, 2) if n else None
        row[f"{q}_flat_%"] = round(100 * in_long[m].mean(), 2) if n else None
    events = [
        {"name": name, "codi_variable": code, "rank": i + 1, "time": t, "flow_30min": v}
        for i, (t, v) in enumerate(top_events(g))
    ]
    return row, events


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat qc-report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--interim", default="data/interim")
    p.add_argument("--net", action="store_true", help="On the clean series (qc-apply)")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    interim = Path(args.interim)
    out = interim / ("qc_net" if args.net else "qc")
    out.mkdir(parents=True, exist_ok=True)
    flat_hours = cfg.qc.flatline_intervals * cfg.resolution_min / 60

    rows, events = [], []
    for name, code in series_list(cfg):
        f = interim / ("aca_flow_qc" if args.net else "aca_flow") / f"{code}.parquet"
        if not f.exists():
            log.warning("%s: missing %s", name, f)
            continue
        log.info("%s", name)
        df = pd.read_parquet(f)
        if args.net:
            df = df.drop(columns="value").rename(columns={"value_qc": "value"})
        r, e = describe(name, code, df, cfg.qc.flatline_min_value, flat_hours)
        rows.append(r)
        events += e
    if not rows:
        raise SystemExit("No series: run llobregat aca-merge first")
    rep = pd.DataFrame(rows)
    ev = pd.DataFrame(events)
    rep.to_csv(out / "report.csv", index=False, encoding="utf-8-sig")
    ev.to_csv(out / "peaks.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\nRange and jumps (m³/s):")
    print(
        rep[
            [
                "name",
                "negatives",
                "zeros_%",
                "p50",
                "p99_9",
                "max_30min",
                "max_time",
                "jump30_p99_9",
                "jump30_max",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nIsolated spikes and flat runs (≥ {flat_hours:g} h with flow > "
        f"{cfg.qc.flatline_min_value:g} m³/s); per 100,000 readings by review status:"
    )
    print(
        rep[
            [
                "name",
                "isolated_spikes",
                "long_flat_runs",
                "longest_flat_run_h",
                "NQ_REV0_%",
                "NQ_REV0_spikes_per_100k",
                "NQ_REV1_spikes_per_100k",
                "NQ_REV0_flat_%",
                "NQ_REV1_flat_%",
            ]
        ].to_string(index=False)
    )
    tgt = ev[ev["codi_variable"] == cfg.target.opendata]
    if len(tgt):
        print(f"\nLargest floods at {cfg.name} (30-min mean):")
        for r in tgt.itertuples():
            print(f"  {r.rank}. {r.time:%Y-%m-%d %H:%M} UTC  {r.flow_30min:,.0f} m³/s")
    print(f"\nSaved to {out}")

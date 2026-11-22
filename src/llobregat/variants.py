"""Model variants: input selection by coverage, and validation folds.

For each variant in the configuration (e.g. "long" since 2000 and "full" since 2018):
  1. compute the coverage of every candidate input (upstream flows, reservoir outflows and
     volumes, XEMA and ACA rain gauges) within the variant's period;
  2. select every series with coverage >= min_coverage;
  3. build the validation folds over the time steps with an observed target.
All variants share the final hold-out (the same test years).

  uv run llobregat splits [--config configs/stations/sant_joan_despi.yaml]

Output in data/interim/variants/: <variant>_inputs.csv and <variant>_folds.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import ModelVariant, StationConfig, load_station_config
from llobregat.ingest.aca_opendata import safe_name
from llobregat.validation import folds_table, hydro_year, make_folds

log = logging.getLogger("variants")

SLOT = pd.Timedelta("30min")
MIN_DAYS_LAST_YEAR = 330  # a hydrological year with fewer target days is left out of the folds


@dataclass
class Candidate:
    kind: str  # flow | outflow | volume | rain_xema | rain_aca
    name: str
    times: pd.DatetimeIndex  # time steps with data (UTC)
    daily: bool = False


# --------------------------------------------------------------------------- #
# Time steps with data, per source
# --------------------------------------------------------------------------- #
def _times_parquet(path: Path) -> pd.DatetimeIndex | None:
    """Time steps with data in a file (value_qc if quality-controlled, else value, or
    volum_embassat for reservoirs)."""
    qc = path.parent.with_name(path.parent.name + "_qc") / path.name
    if path.parent.name == "aca_flow" and qc.exists():
        path = qc
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    col = next((c for c in ("value_qc", "value", "volum_embassat") if c in df.columns), None)
    if col is not None:
        df = df[df[col].notna()]
    return pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)).sort_values()


def _times_xema(root: Path, station: str) -> pd.DatetimeIndex | None:
    d = root / "readings" / "variable=35" / f"station={station}"
    files = sorted(d.glob("year=*.parquet"))
    if not files:
        return None
    df = pd.concat(
        [pd.read_parquet(f, columns=["data_lectura", "codi_base"]) for f in files],
        ignore_index=True,
    )
    t = pd.to_datetime(df["data_lectura"], utc=True)
    hourly = df["codi_base"].eq("HO").to_numpy()
    # An hourly reading covers both half hours
    t = pd.DatetimeIndex(np.concatenate([t.to_numpy(), (t[hourly] + SLOT).to_numpy()]))
    return t.tz_localize("UTC") if t.tz is None else t


def gather_candidates(cfg: StationConfig, interim: Path, xema_root: Path) -> list[Candidate]:
    out: list[Candidate] = []
    for g in cfg.upstream:
        if g.aca.opendata:
            t = _times_parquet(interim / "aca_flow" / f"{g.aca.opendata}.parquet")
            out.append(Candidate("flow", g.name, t if t is not None else pd.DatetimeIndex([])))
    for r in cfg.reservoirs:
        for kind, ref in (("outflow", r.outflow), ("volume", r.volume)):
            if ref is not None and ref.opendata_daily:
                f = interim / "aca_daily" / f"{safe_name(r.name)}__{kind}.parquet"
                t = _times_parquet(f)
                out.append(
                    Candidate(
                        kind, r.name, t if t is not None else pd.DatetimeIndex([]), daily=True
                    )
                )
    for st in cfg.rain.xema_stations:
        times = [_times_xema(xema_root, st)]
        old = cfg.rain.splices.get(st)
        if old:
            times.append(_times_xema(xema_root, old))
        times = [t for t in times if t is not None]
        t = times[0].append(times[1:]) if times else pd.DatetimeIndex([])
        out.append(Candidate("rain_xema", st, t))
    for code in cfg.rain.aca_gauges:
        t = _times_parquet(interim / "aca_rain" / f"{code}.parquet")
        out.append(Candidate("rain_aca", code, t if t is not None else pd.DatetimeIndex([])))
    return out


# --------------------------------------------------------------------------- #
# Coverage and selection
# --------------------------------------------------------------------------- #
def coverage(
    times: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, daily: bool = False
) -> float:
    """Fraction of intervals (30 min, or days for daily series) with data."""
    step = pd.Timedelta("1D") if daily else SLOT
    if len(times) == 0 or end <= start:
        return 0.0
    t = times[(times >= start) & (times < end)]
    slots = t.floor("1D" if daily else "30min").unique()
    return float(min(len(slots) / ((end - start) / step), 1.0))


def select_inputs(cands: list[Candidate], variant: ModelVariant, end: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(variant.train_start, tz="UTC")
    rows = [
        {
            "kind": c.kind,
            "name": c.name,
            "coverage": round(coverage(c.times, start, end, c.daily), 3),
        }
        for c in cands
    ]
    df = pd.DataFrame(rows)
    df["selected"] = (df["coverage"] >= variant.min_coverage).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Target index and folds
# --------------------------------------------------------------------------- #
def target_index(cfg: StationConfig, interim: Path) -> pd.DatetimeIndex:
    """Time steps of the 30-min grid with an observed target flow."""
    f = interim / "aca_flow_qc" / f"{cfg.target.opendata}.parquet"
    col = "value_qc"
    if not f.exists():
        f, col = interim / "aca_flow" / f"{cfg.target.opendata}.parquet", "value"
        log.warning("No quality control: run llobregat qc-apply")
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat aca-merge first")
    df = pd.read_parquet(f, columns=["timestamp", col]).rename(columns={col: "value"}).dropna()
    s = df.set_index("timestamp")["value"]
    grid = s.resample("30min", label="right", closed="right").mean().dropna()
    return pd.DatetimeIndex(grid.index)


def complete_years_only(index: pd.DatetimeIndex, start_month: int) -> pd.DatetimeIndex:
    """Drop the last hydrological year if it does not have enough days yet (current year)."""
    years = hydro_year(index, start_month)
    last = years.max()
    days = pd.DatetimeIndex(index[years == last]).normalize().nunique()
    if days < MIN_DAYS_LAST_YEAR:
        log.info("Hydrological year %d incomplete (%d days): left out of the folds", last, days)
        return index[years != last]
    return index


def variant_folds(
    cfg: StationConfig, variant: ModelVariant, index: pd.DatetimeIndex
) -> pd.DataFrame:
    start = pd.Timestamp(variant.train_start, tz="UTC")
    idx = index[index >= start]
    vcfg = cfg.validation_for(variant)
    idx = complete_years_only(idx, vcfg.hydro_year_start_month)
    table = folds_table(make_folds(idx, vcfg))
    table.insert(0, "variant", variant.name)
    return table


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat splits",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--interim", default="data/interim")
    p.add_argument("--xema", default="data/raw/xema")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    if not cfg.variants:
        raise SystemExit("The configuration has no variants")
    interim = Path(args.interim)
    out = interim / "variants"
    out.mkdir(parents=True, exist_ok=True)

    index = target_index(cfg, interim)
    end = index.max() + SLOT
    cands = gather_candidates(cfg, interim, Path(args.xema))
    missing = [f"{c.kind}:{c.name}" for c in cands if len(c.times) == 0]
    if missing:
        log.warning("No data (or missing file): %s", ", ".join(missing))

    pd.set_option("display.width", 200)
    sel_all, folds_all = [], []
    for v in cfg.variants:
        sel = select_inputs(cands, v, end)
        sel.insert(0, "variant", v.name)
        sel.to_csv(out / f"{v.name}_inputs.csv", index=False, encoding="utf-8-sig")
        folds = variant_folds(cfg, v, index)
        folds.to_csv(out / f"{v.name}_folds.csv", index=False, encoding="utf-8-sig")
        sel_all.append(sel)
        folds_all.append(folds)
        print(f"\n=== Variant {v.name}: {v.description or ''} (from {v.train_start}) ===")
        for kind, part in sel.groupby("kind", sort=False):
            inn = part[part["selected"] == 1]
            outn = part[part["selected"] == 0]
            print(
                f"  {kind:<10} selected {len(inn):>2}/{len(part):<2}: "
                + ", ".join(f"{r.name} ({r.coverage:.0%})" for r in inn.itertuples())
            )
            if len(outn):
                print(
                    " " * 13
                    + "out: "
                    + ", ".join(f"{r.name} ({r.coverage:.0%})" for r in outn.itertuples())
                )
        print(folds.drop(columns="variant").to_string(index=False))

    # Check: all variants must share the hold-out
    finals = {f["variant"].iloc[0]: f.loc[f["fold"] == "final", "test"].iloc[0] for f in folds_all}
    if len(set(finals.values())) > 1:
        log.warning("The variants do not share the hold-out: %s", finals)
    else:
        print(f"\nCommon hold-out (hydrological years): {next(iter(finals.values()))}")
    common = set.intersection(*[set(f.loc[f["fold"] != "final", "test"]) for f in folds_all])
    if common:
        print(f"Shared test folds: {', '.join(sorted(common))}")
    print(f"Saved to {out}")

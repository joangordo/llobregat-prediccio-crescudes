"""Merge the ACA open-data history into one series per station.

The 5-min flows come in three datasets with the same codi_variable:
  cabals_1996_2009 (mmzz-keb8) + cabals_2010_2019 (9evs-b5nk) + cabals (3yr3-vq6y, 2020-)
This step chains them (where they overlap, the most recent dataset wins), saves one series
per station and reports the coverage per hydrological year (October-September), measured as
the fraction of 30-min intervals with at least one reading. The hydrological year is
labelled with the year in which it ends (2020 = October 2019 - September 2020). It also
chains the configured daily series (dam outflows, reservoir volumes and long daily flows).

  uv run llobregat aca-merge [--config configs/stations/sant_joan_despi.yaml]

Output in data/interim/:
  aca_flow/<codi_variable>.parquet      timestamp (UTC), value, quality_level, source
  aca_flow/coverage.csv                 station x hydrological year
  aca_flow/quality.csv                  quality levels per station and source dataset
  aca_flow/units.csv                    stretches in L/s detected and corrected
  aca_daily/<name>__<kind>.parquet      chained daily series

Units: some 5-min stretches are in L/s even though the dataset says m³/s. They are detected
by comparing each day with the official daily flow (cabals_diaris, always in m³/s) and
divided by 1000; the original value is kept in value_raw.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import SensorRef, StationConfig, load_station_config
from llobregat.ingest.aca_opendata import safe_name
from llobregat.validation import hydro_year as _hy

log = logging.getLogger("aca_merge")

SEGMENTS = ["cabals_1996_2009", "cabals_2010_2019", "cabals"]  # the later one wins
MAX_DAILY_M3S = 5000.0  # no daily flow in the basin gets there (Gloria at SJD: ~1,000)
SLOT = "30min"


# --------------------------------------------------------------------------- #
# Inventory of series from the configuration
# --------------------------------------------------------------------------- #
def configured_series(cfg: StationConfig) -> list[tuple[str, str, SensorRef]]:
    """(name, kind, sensor) of every series in the configuration."""
    out = [(f"{cfg.station_id} (target)", "flow", cfg.target)]
    out += [(g.name, "flow", g.aca) for g in cfg.upstream]
    for r in cfg.reservoirs:
        for kind in ("outflow", "volume", "level"):
            ref = getattr(r, kind)
            if ref is not None:
                out.append((r.name, kind, ref))
    for s in cfg.log_only:
        out.append(((s.note or s.sensor).split(" (")[0][:40], s.variable, s))
    return out


# --------------------------------------------------------------------------- #
# Chaining
# --------------------------------------------------------------------------- #
def merge_segments(raw: Path, code: str, segments: list[str] = SEGMENTS) -> pd.DataFrame:
    parts = []
    for rank, seg in enumerate(segments):
        d = raw / "readings" / f"dataset={seg}" / f"station={safe_name(code)}"
        for f in sorted(d.glob("year=*.parquet")):
            df = pd.read_parquet(f)
            if df.empty:
                continue
            df["source"] = seg
            df["_rank"] = rank
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "value", "quality_level", "source"])
    df = pd.concat(parts, ignore_index=True).rename(columns={"nivell_qualitat": "quality_level"})
    df = df.sort_values(["timestamp", "_rank"]).drop_duplicates("timestamp", keep="last")
    cols = [c for c in ("timestamp", "value", "quality_level", "source") if c in df.columns]
    return df[cols].reset_index(drop=True)


def merge_daily(raw: Path, dataset: str, keys: list[str]) -> pd.DataFrame:
    parts = []
    for rank, key in enumerate(keys):
        f = raw / "readings" / f"dataset={dataset}" / f"station={safe_name(key)}" / "all.parquet"
        if not f.exists():
            log.warning("Missing daily series %s (%s)", key, f)
            continue
        df = pd.read_parquet(f)
        df["_rank"] = rank
        df["source"] = key
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if dataset == "cabals_diaris" and "value" in df:
        # Impossible values in the official daily series (e.g. 3,369,195 m³/s at Aigua de Valls)
        bad = (df["value"] < 0) | (df["value"] > MAX_DAILY_M3S)
        if bad.any():
            log.warning("%s: %d days with impossible daily flow (discarded)", keys, int(bad.sum()))
            df.loc[bad, "value"] = np.nan
    df["day"] = df["timestamp"].dt.floor("D")
    df = df.sort_values(["day", "_rank"]).drop_duplicates("day", keep="last")
    return df.drop(columns=["_rank", "day"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Units: some 5-min stretches are in L/s even though the dataset says m³/s
# --------------------------------------------------------------------------- #
UNIT_TOL = 0.7  # log10 tolerance around x1 (m³/s) and x1000 (L/s)
MIN_DAILY_M3S = 0.01  # below this, the ratio is not reliable (dry river)


def unit_factors(df5: pd.DataFrame, daily: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Factor per reading (1 or 0.001), decided for whole days.

    For each day, q = log10(mean of the day's readings / official daily flow in m³/s): close
    to 0 if the day is in m³/s and close to 3 if it is in L/s. Means are compared with means
    (not reading by reading) because during a flood the instantaneous flow can be tens of
    times the daily mean. Days without a clear class (dry river, no daily value, odd ratio)
    take the majority unit of the ±15 surrounding days or, failing that, that of the nearest
    classified day. A unit change in the middle of a day leaves half a day wrong: the quality
    control (daily balance) catches it afterwards.
    """
    t = pd.DatetimeIndex(df5["timestamp"])
    day = t.floor("D")
    x = df5["value"].to_numpy(dtype=float)
    m5 = pd.Series(x, index=day).groupby(level=0).mean()
    ref = pd.Series(
        daily["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(daily["timestamp"]).floor("D")
    )
    ref = ref[~ref.index.duplicated(keep="last")].reindex(m5.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.log10(m5 / ref).where((ref > MIN_DAILY_M3S) & (m5 > 0))
    cls = pd.Series("no_ref", index=m5.index, dtype=object)
    cls[(q - 0).abs() < UNIT_TOL] = "m3s"
    cls[(q - 3).abs() < UNIT_TOL] = "ls"
    cls[(q >= UNIT_TOL) & (q <= 3 - UNIT_TOL)] = "mixed"
    cls[(q < -UNIT_TOL) | (q > 3 + UNIT_TOL)] = "odd"
    own = pd.Series(np.nan, index=m5.index)
    own[cls == "m3s"] = 0.0
    own[cls == "ls"] = 3.0
    full = pd.date_range(m5.index.min(), m5.index.max(), freq="D")
    neigh = own.reindex(full).rolling(31, center=True, min_periods=3).median().reindex(m5.index)
    lvl = own.fillna(neigh).ffill().bfill().fillna(0.0)
    fday = pd.Series(np.where(lvl >= 1.5, 0.001, 1.0), index=m5.index)
    f = fday.reindex(day).to_numpy()
    summary = pd.DataFrame({"class": cls, "day_factor": fday, "log10_ratio": q})
    return f, summary


def apply_units(df5: pd.DataFrame, daily: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    if daily is None or daily.empty:
        return df5.assign(unit_factor=1.0), {"reference": "none"}
    f, summ = unit_factors(df5, daily)
    out = df5.assign(value_raw=df5["value"], unit_factor=f)
    out["value"] = out["value_raw"] * out["unit_factor"]
    years = pd.DatetimeIndex(out["timestamp"]).year
    ls_share = pd.Series(f < 1, index=years).groupby(level=0).mean()
    shares = summ["class"].value_counts(normalize=True)
    info = {
        "reference": "daily",
        "readings_ls_%": round(100 * float((f < 1).mean()), 1),
        **{
            f"days_{k}_%": round(100 * float(shares.get(k, 0)), 1)
            for k in ("m3s", "ls", "mixed", "odd", "no_ref")
        },
        "years_mostly_ls": compact_years([int(y) for y, s in ls_share.items() if s > 0.5]),
    }
    return out, info


def compact_years(years: list[int]) -> str:
    """[2001, 2002, 2003, 2007] -> '2001-2003, 2007'."""
    if not years:
        return ""
    out, a, b = [], years[0], years[0]
    for y in years[1:]:
        if y == b + 1:
            b = y
        else:
            out.append(f"{a}-{b}" if b > a else str(a))
            a = b = y
    out.append(f"{a}-{b}" if b > a else str(a))
    return ", ".join(out)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def hydro_year(ts: pd.Series | pd.DatetimeIndex, start_month: int = 10) -> np.ndarray:
    """Hydrological year labelled with the year in which it ends (Oct 2019 - Sep 2020 ->
    2020), as in llobregat.validation."""
    return _hy(pd.DatetimeIndex(ts), start_month)


def coverage_by_hydro_year(
    ts: pd.Series, years: list[int], until: pd.Timestamp, start_month: int = 10
) -> pd.Series:
    """Fraction of 30-min intervals of each hydrological year with at least one reading."""
    slots = pd.Series(pd.DatetimeIndex(ts).floor(SLOT).unique())
    have = pd.Series(hydro_year(slots, start_month)).value_counts()
    out = {}
    for y in years:
        a = pd.Timestamp(year=y - 1, month=start_month, day=1, tz="UTC")
        b = min(pd.Timestamp(year=y, month=start_month, day=1, tz="UTC"), until)
        total = max((b - a) / pd.Timedelta(SLOT), 1)
        out[y] = min(have.get(y, 0) / total, 1.0)
    return pd.Series(out)


def heat_char(x: float) -> str:
    if x >= 0.9:
        return "█"
    if x >= 0.5:
        return "▓"
    if x > 0:
        return "░"
    return "·"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat aca-merge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--raw", default="data/raw/aca_opendata")
    p.add_argument("--out", default="data/interim")
    p.add_argument(
        "--no-units-fix",
        action="store_true",
        help="Do not correct the stretches in L/s (for comparison only)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    raw, out = Path(args.raw), Path(args.out)
    (out / "aca_flow").mkdir(parents=True, exist_ok=True)
    (out / "aca_daily").mkdir(parents=True, exist_ok=True)
    series = configured_series(cfg)
    start_month = cfg.validation.hydro_year_start_month

    # 1) Chained daily series (they also serve as the unit reference)
    rows, daily_by_code = [], {}
    for name, kind, ref in series:
        if not ref.opendata_daily:
            continue
        ds = "embassaments_diaris" if kind in ("volume", "level") else "cabals_diaris"
        df = merge_daily(raw, ds, ref.opendata_daily)
        if df.empty:
            rows.append((name, kind, None, None, 0))
            continue
        df.to_parquet(out / "aca_daily" / f"{safe_name(name)}__{kind}.parquet", index=False)
        if kind == "flow" and ref.opendata:
            daily_by_code[ref.opendata] = df
        rows.append(
            (name, kind, df["timestamp"].min().date(), df["timestamp"].max().date(), len(df))
        )

    # 2) Chained 5-min series, with corrected units
    merged: dict[str, tuple[str, pd.DataFrame]] = {}
    unit_rows = []
    for name, kind, ref in series:
        if not ref.opendata or kind not in ("flow", "level") or ref.opendata in merged:
            continue
        df = merge_segments(raw, ref.opendata)
        if df.empty:
            log.warning("%s (%s): no 5-min data", name, ref.opendata)
            continue
        if kind == "flow" and not args.no_units_fix:
            df, info = apply_units(df, daily_by_code.get(ref.opendata))
            unit_rows.append({"name": name, "codi_variable": ref.opendata, **info})
        df.to_parquet(out / "aca_flow" / f"{ref.opendata}.parquet", index=False)
        merged[ref.opendata] = (f"{name} · {kind}", df)
    if not merged:
        raise SystemExit(f"No readings in {raw / 'readings'}: run aca-opendata data first")
    if unit_rows:
        units_df = pd.DataFrame(unit_rows)
        units_df.to_csv(out / "aca_flow" / "units.csv", index=False, encoding="utf-8-sig")

    until = max(df["timestamp"].max() for _, df in merged.values())
    first = min(df["timestamp"].min() for _, df in merged.values())
    years = list(
        range(
            int(hydro_year(pd.DatetimeIndex([first]), start_month)[0]),
            int(hydro_year(pd.DatetimeIndex([until]), start_month)[0]) + 1,
        )
    )
    cov = pd.DataFrame(
        {
            code: coverage_by_hydro_year(df["timestamp"], years, until, start_month)
            for code, (_, df) in merged.items()
        }
    ).T
    cov.index.name = "codi_variable"
    cov.insert(0, "name", [merged[c][0] for c in cov.index])
    cov.round(3).to_csv(out / "aca_flow" / "coverage.csv", encoding="utf-8-sig")

    qual = pd.concat(
        [
            df.assign(codi_variable=code)
            .groupby(["codi_variable", "source"], dropna=False)["quality_level"]
            .value_counts(dropna=False)
            .rename("rows")
            .reset_index()
            for code, (_, df) in merged.items()
            if "quality_level" in df.columns
        ],
        ignore_index=True,
    )
    qual.to_csv(out / "aca_flow" / "quality.csv", index=False, encoding="utf-8-sig")

    # Coverage map in the console
    w = max(len(n) for n, _ in merged.values())
    print(
        f"\nCoverage per hydrological year (█ ≥90 %  ▓ ≥50 %  ░ >0  · none), {years[0]}-{years[-1]}"
    )
    marks = [" "] * len(years)
    for i, y in enumerate(years):
        if y % 10 == 0 and i + 4 <= len(years):
            marks[i : i + 4] = list(str(y))
    print(" " * (w + 2) + "".join(marks))
    print(" " * (w + 2) + "".join(str(y)[-1] for y in years))
    for code in cov.index:
        row = "".join(heat_char(cov.loc[code, y]) for y in years)
        ok = int((cov.loc[code, years] >= 0.9).sum())
        print(f"{cov.loc[code, 'name']:<{w}}  {row}  {ok:>2} years ≥90 %")
    nq = qual.groupby("quality_level", dropna=False)["rows"].sum()
    print("Quality levels:", {str(k): int(v) for k, v in nq.items()})

    if rows:
        daily = pd.DataFrame(rows, columns=["name", "kind", "start", "end", "days"])
        daily.to_csv(out / "aca_daily" / "summary.csv", index=False, encoding="utf-8-sig")
        print("\nDaily series:")
        print(daily.to_string(index=False))
    if unit_rows:
        pd.set_option("display.width", 250)
        print("\nUnits (daily comparison with the official daily flow, in m³/s):")
        print(units_df.drop(columns="codi_variable").to_string(index=False))
    print(f"\nSaved to {out / 'aca_flow'} and {out / 'aca_daily'}")

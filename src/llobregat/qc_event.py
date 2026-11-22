"""Inspect one episode of a flow series (to review suspicious values).

Around the maximum of a day, it shows the readings aggregated to 30 min (count, minimum,
mean and maximum, original value before unit correction, unit factor, quality level and
source) and the official daily flow of the surrounding days.

  uv run llobregat qc-event --code 4652845 --date 2024-04-29
  uv run llobregat qc-event --maxima        (the maximum of each station, to a file)

Output of --maxima: data/interim/qc/maxima.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llobregat.config import load_station_config
from llobregat.ingest.aca_merge import configured_series
from llobregat.ingest.aca_opendata import safe_name


def daily_file(interim: Path, cfg, code: str) -> Path | None:
    for name, kind, ref in configured_series(cfg):
        if ref.opendata == code and kind == "flow":
            return interim / "aca_daily" / f"{safe_name(name)}__flow.parquet"
    return None


def name_of(cfg, code: str) -> str:
    for name, kind, ref in configured_series(cfg):
        if ref.opendata == code:
            return f"{name} · {kind}"
    return code


def episode(
    df: pd.DataFrame, daily: pd.DataFrame | None, center: pd.Timestamp, hours: float
) -> str:
    half = pd.Timedelta(hours=hours / 2)
    w = df[(df["timestamp"] >= center - half) & (df["timestamp"] <= center + half)].copy()
    if w.empty:
        return "  (no readings in this window)\n"
    w["slot"] = w["timestamp"].dt.ceil("30min")
    agg = {
        "n": ("value", "size"),
        "min": ("value", "min"),
        "mean": ("value", "mean"),
        "max": ("value", "max"),
    }
    if "value_raw" in w:
        agg["max_raw"] = ("value_raw", "max")
    if "unit_factor" in w:
        agg["factor"] = ("unit_factor", lambda s: "/".join(sorted({f"{x:g}" for x in s})))
    if "quality_level" in w:
        agg["quality"] = ("quality_level", lambda s: "/".join(sorted(set(map(str, s)))))
    if "source" in w:
        agg["source"] = ("source", lambda s: "/".join(sorted(set(map(str, s)))))
    if "value_qc" in w:
        agg["max_clean"] = ("value_qc", "max")
        agg["flags"] = (
            "flags",
            lambda s: "/".join(sorted({m for x in s for m in str(x).split(",") if m})),
        )
    tab = w.groupby("slot").agg(**agg)
    tab.index = tab.index.strftime("%Y-%m-%d %H:%M")
    for c in ("min", "mean", "max", "max_raw", "max_clean"):
        if c in tab:
            tab[c] = tab[c].round(2)
    txt = tab.to_string() + "\n"
    if daily is not None and not daily.empty:
        days = pd.date_range(
            (center - pd.Timedelta(days=1)).floor("D"),
            (center + pd.Timedelta(days=1)).floor("D"),
            freq="D",
        )
        d = daily.assign(day=pd.DatetimeIndex(daily["timestamp"]).floor("D")).set_index("day")
        vals = [f"{x:%Y-%m-%d}: {d['value'].get(x, float('nan')):.2f}" for x in days]
        txt += "  Official daily flow (m³/s): " + " · ".join(vals) + "\n"
    return txt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat qc-event",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--interim", default="data/interim")
    p.add_argument("--code", help="codi_variable of the series (e.g. 4652845)")
    p.add_argument("--date", help="Day of the episode (YYYY-MM-DD); centred on the day's maximum")
    p.add_argument("--hours", type=float, default=6, help="Window width (hours)")
    p.add_argument("--maxima", action="store_true", help="The maximum of each station")
    p.add_argument(
        "--net", action="store_true", help="With --maxima: the maximum after quality control"
    )
    p.add_argument("--no-qc", action="store_true", help="Ignore data/interim/aca_flow_qc")
    p.add_argument(
        "--rule",
        help="With --code: the longest episodes flagged by a rule "
        "(spike, excursion, oscillation, stuck, high_flat, daily_balance...)",
    )
    args = p.parse_args(argv)
    cfg = load_station_config(args.config)
    interim = Path(args.interim)

    def load(code: str):
        qc = interim / "aca_flow_qc" / f"{code}.parquet"
        src = qc if (qc.exists() and not args.no_qc) else interim / "aca_flow" / f"{code}.parquet"
        df = pd.read_parquet(src)
        f = daily_file(interim, cfg, code)
        daily = pd.read_parquet(f) if f is not None and f.exists() else None
        return df, daily

    if args.maxima:
        out = []
        for f in sorted((interim / "aca_flow").glob("*.parquet")):
            code = f.stem
            df, daily = load(code)
            col = "value_qc" if (args.net and "value_qc" in df) else "value"
            s = df.set_index("timestamp")[col].dropna()
            center = s.idxmax()
            out.append(
                f"=== {name_of(cfg, code)} ({code}) · maximum {s.max():,.1f} m³/s "
                f"on {center:%Y-%m-%d %H:%M} UTC ===\n"
            )
            out.append(episode(df, daily, center, args.hours) + "\n")
        path = interim / "qc" / ("maxima_clean.txt" if args.net else "maxima.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(out), encoding="utf-8")
        print("".join(out[:4]))
        print(f"... saved in full to {path}")
        return

    if args.code and args.rule:
        df, daily = load(args.code)
        if "flags" not in df:
            raise SystemExit("Run llobregat qc-apply first")
        m = df["flags"].astype(str).str.contains(args.rule, regex=False).to_numpy()
        if not m.any():
            raise SystemExit(f"No reading flagged with {args.rule}")
        gid = (pd.Series(m) != pd.Series(m).shift()).cumsum()
        ep = (
            df[m]
            .assign(g=gid[m].to_numpy())
            .groupby("g")
            .agg(
                start=("timestamp", "min"),
                end=("timestamp", "max"),
                readings=("value", "size"),
                minimum=("value", "min"),
                maximum=("value", "max"),
            )
        )
        ep["hours"] = ((ep["end"] - ep["start"]) / pd.Timedelta(hours=1)).round(1)
        ep = ep.sort_values("hours", ascending=False)
        print(
            f"=== {name_of(cfg, args.code)} · rule {args.rule}: {len(ep)} episodes, "
            f"{int(m.sum())} readings ==="
        )
        show = ep.head(8).copy()
        show["start"] = show["start"].dt.strftime("%Y-%m-%d %H:%M")
        show["end"] = show["end"].dt.strftime("%Y-%m-%d %H:%M")
        print(show.round(2).to_string(index=False))
        top = ep.iloc[0]
        print(f"\nStart of the longest episode ({top['hours']} h):")
        print(episode(df, daily, top["start"] + pd.Timedelta(hours=args.hours / 2 - 1), args.hours))
        return

    if not (args.code and args.date):
        raise SystemExit("Give --code and --date, --code and --rule, or --maxima")
    df, daily = load(args.code)
    day = pd.Timestamp(args.date, tz="UTC")
    s = df.set_index("timestamp")["value"]
    sd = s[(s.index >= day) & (s.index < day + pd.Timedelta(days=1))]
    if sd.empty:
        raise SystemExit(f"No readings on {args.date}")
    center = sd.idxmax()
    print(
        f"=== {name_of(cfg, args.code)} · maximum of the day {sd.max():,.1f} m³/s "
        f"at {center:%H:%M} UTC ==="
    )
    print(episode(df, daily, center, args.hours))

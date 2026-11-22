"""Project command line.

uv run llobregat --help
uv run llobregat config check
uv run llobregat xema <command> ...          (XEMA open data download)
uv run llobregat aca catalog [--component-type TYPE]
uv run llobregat aca sensor PROVIDER SENSOR [--hours 24]
uv run llobregat aca rain-map               (real-time sensor of each historical rain gauge)
uv run llobregat realtime [--store data/realtime]
uv run llobregat aca-opendata <command> ... (ACA flow and reservoir history, 1996-)
uv run llobregat aca-rain [--basins ...]     (ACA rain gauge history, Excel files)
uv run llobregat aca-merge                   (merge the ACA history and report its coverage)
uv run llobregat qc-report                   (quality report of the historical flows)
uv run llobregat qc-event --maxima           (detail of the maxima or of one episode)
uv run llobregat qc-apply                    (quality control of the 5-min flows)
uv run llobregat splits                      (inputs and validation folds of each variant)
uv run llobregat build-dataset               (30-min data table of each variant)
uv run llobregat analyse-features            (correlations, duplicates and travel times)
uv run llobregat train --variant long        (baselines, Ridge and LightGBM, yearly folds)
uv run llobregat compare --variant long      (model table and statistical comparison)
uv run llobregat combine --variant long --models lgbm ridge   (mean of trained models)
uv run llobregat tune --variant long         (LightGBM hyperparameters with Optuna)
uv run llobregat intervals --variant long    (quantiles, CQR, ACI and warnings, yearly folds)
uv run llobregat final-evaluation --variant long   (hold-out 2025-2026: once only)
uv run llobregat report --variant long       (final report: metrics, lead times, figures)
uv run llobregat operational-model --variant long  (retrain with all the data for the app)
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from llobregat.config import load_all_stations

log = logging.getLogger("llobregat")

# Commands with their own argument parser: name -> (module, function)
ROUTES = {
    "intervals": ("llobregat.probabilistic", "main"),
    "final-evaluation": ("llobregat.probabilistic", "main_final"),
    "report": ("llobregat.report", "main"),
    "operational-model": ("llobregat.operational", "main"),
    "tune": ("llobregat.tuning", "main"),
    "analyse-features": ("llobregat.analysis", "main"),
    "train": ("llobregat.train", "main"),
    "compare": ("llobregat.train", "main_compare"),
    "combine": ("llobregat.train", "main_combine"),
    "build-dataset": ("llobregat.dataset", "main"),
    "qc-report": ("llobregat.qc_report", "main"),
    "qc-event": ("llobregat.qc_event", "main"),
    "qc-apply": ("llobregat.qc_flow", "main"),
    "splits": ("llobregat.variants", "main"),
    "aca-merge": ("llobregat.ingest.aca_merge", "main"),
    "aca-rain": ("llobregat.ingest.aca_rain_history", "main"),
    "aca-opendata": ("llobregat.ingest.aca_opendata", "main"),
}


def cmd_config_check(args: argparse.Namespace) -> int:
    configs = load_all_stations(args.configs)
    if not configs:
        print(f"No configuration in {args.configs}")
        return 1
    from llobregat.ingest.realtime import aca_series

    for sid, cfg in configs.items():
        todo = []
        if "TODO" in (cfg.target.provider, cfg.target.sensor):
            todo.append("target")
        todo += [u.name for u in cfg.upstream if "TODO" in (u.aca.provider, u.aca.sensor)]
        status = "OK" if not todo else f"pending: {', '.join(todo)}"
        print(f"{sid}: {cfg.name} · horizons {cfg.horizons_h} h · {status}")
        print(
            f"  upstream gauges: {len(cfg.upstream)} · reservoirs: "
            f"{len(cfg.reservoirs)} · logged only: {len(cfg.log_only)}"
        )
        print(
            f"  XEMA stations: {len(cfg.rain.xema_stations)} · sub-basins: "
            f"{len(cfg.rain.subbasins)} · splices: {cfg.rain.splices}"
        )
    n = len(aca_series(list(configs.values())))
    print(f"ACA series logged in real time: {n}")
    return 0


def cmd_aca_catalog(args: argparse.Namespace) -> int:
    from llobregat.ingest.aca_sentilo import SentiloClient, flatten_catalog

    df = flatten_catalog(SentiloClient().catalog(args.component_type))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} sensors saved to {out}")
    if not df.empty:
        print(df.groupby(["component_type", "type"], dropna=False).size().to_string())
    return 0


def cmd_aca_sensor(args: argparse.Namespace) -> int:
    from llobregat.ingest.aca_sentilo import SentiloClient, observations_to_frame

    now = datetime.now(UTC)
    obs = SentiloClient().observations_range(
        args.provider, args.sensor, start=now - timedelta(hours=args.hours), end=now
    )
    df = observations_to_frame(obs, args.provider, args.sensor, "raw")
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    print(df.tail(args.show).to_string(index=False))
    print(f"\n{len(df)} observations in the last {args.hours} h")
    if obs:
        print("Example of a raw observation:", obs[-1])
    return 0


def _norm(text: str) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return " ".join(t.replace("_", " ").replace("-", " ").split())


def cmd_aca_rain_map(args: argparse.Namespace) -> int:
    """Propose the real-time sensor of each historical ACA rain gauge.

    The Sentilo catalogue does not name the site of its rain gauges (every description is
    "Intensitat de precipitació"), so the historical gauge names (e.g. "Cercs_Baells") are
    matched with the sites of the rain gauges listed in log_only (e.g. "Cercs (la Baells)").
    """
    import pandas as pd

    from llobregat.config import load_station_config

    stop = {"de", "del", "dels", "la", "el", "les", "els", "i", "d", "l", "etap"}

    def tokens(text: str) -> set[str]:
        return {t for t in _norm(text).replace("'", " ").replace("(", " ").split() if t not in stop}

    cfg = load_station_config(args.config)
    st = pd.read_csv(args.stations, dtype=str)
    st = st[st["code"].isin(cfg.rain.aca_gauges)]
    sites = [s for s in cfg.log_only if "PLUVIOMETRE" in s.provider and s.note]
    lines = ["  aca_realtime:"]
    for r in st.itertuples():
        name = tokens(r.name.replace(")", " "))
        scored = sorted(
            ((len(name & tokens(s.note.replace(")", " "))) / max(len(name), 1), s) for s in sites),
            key=lambda x: -x[0],
        )
        (best, s1), (second, s2) = scored[0], scored[1]
        sure = best >= 0.5 and best > second
        print(
            f"{r.code:<7} {r.name:<32} -> {s1.sensor:<20} {s1.note}  ({best:.2f}"
            + ("" if sure else f"; CHECK: next {s2.note} {second:.2f}")
            + ")"
        )
        sensor, note = (s1.sensor, s1.note) if sure else ("TODO", "no matching site")
        lines.append(f"    {r.code}: {{provider: {s1.provider}, sensor: {sensor}}}  # {note}")
    print("\nFor the station YAML (rain section):\n" + "\n".join(lines))
    return 0


def cmd_realtime(args: argparse.Namespace) -> int:
    from llobregat.ingest.realtime import run_realtime

    configs = list(load_all_stations(args.configs).values())
    result = run_realtime(configs, args.store)
    print(result)
    return 0 if all(v >= 0 for v in result.values()) else 1


def cmd_xema(argv: list[str]) -> int:
    from llobregat.ingest import xema_opendata

    xema_opendata.main(argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llobregat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--configs", default="configs/stations", help="Configuration folder")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("config", help="Station configurations")
    csub = c.add_subparsers(dest="config_cmd", required=True)
    csub.add_parser("check", help="Validate the YAML files and show what is missing").set_defaults(
        fn=cmd_config_check
    )

    a = sub.add_parser("aca", help="ACA real-time API (Sentilo)")
    asub = a.add_subparsers(dest="aca_cmd", required=True)
    cat = asub.add_parser("catalog", help="Save the sensor catalogue")
    cat.add_argument("--component-type", default=None)
    cat.add_argument("--out", default="data/raw/aca/catalog.csv")
    cat.set_defaults(fn=cmd_aca_catalog)
    sen = asub.add_parser("sensor", help="Show the latest observations of a sensor")
    sen.add_argument("provider")
    sen.add_argument("sensor")
    sen.add_argument("--hours", type=float, default=24)
    sen.add_argument("--show", type=int, default=10)
    sen.set_defaults(fn=cmd_aca_sensor)
    rm = asub.add_parser(
        "rain-map", help="Propose the real-time sensor of each historical rain gauge"
    )
    rm.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    rm.add_argument("--stations", default="data/interim/aca_rain/stations.csv")
    rm.set_defaults(fn=cmd_aca_rain_map)

    r = sub.add_parser("realtime", help="Log the new ACA and Meteocat data")
    r.add_argument("--store", default="data/realtime")
    r.set_defaults(fn=cmd_realtime)

    sub.add_parser(
        "xema",
        help="XEMA download (open data); see 'llobregat xema -h'",
        add_help=False,
    )
    for name in ROUTES:
        sub.add_parser(name, help=f"see 'llobregat {name} -h'", add_help=False)
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # These commands pass the rest of the arguments to their own parser
    if argv and argv[0] in ROUTES:
        module, func = ROUTES[argv[0]]
        getattr(importlib.import_module(module), func)(argv[1:])
        sys.exit(0)
    if "xema" in argv:
        i = argv.index("xema")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        sys.exit(cmd_xema(argv[i + 1 :]))
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()

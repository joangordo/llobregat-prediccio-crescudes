"""Download of the ACA history from the Catalan government open data portal.

Datasets (Socrata API, like XEMA). According to the ACA, all times are UTC+0. The dataset
names are those of the portal (in Catalan):

  name                 id         content
  cabals               3yr3-vq6y  flows, logging interval (5 min), 2020-
  cabals_2010_2019     9evs-b5nk  flows, logging interval, 2010-2019
  cabals_1996_2009     mmzz-keb8  flows, logging interval, 1996-2009
  embassaments         vjx7-6kcp  reservoir level, volume and %, logging interval
  cabals_diaris        2bwe-vgpr  mean daily flow (includes dam outflows), 1951-
  embassaments_diaris  gn9e-3qhr  daily reservoir status
An identifier is also accepted directly (--dataset hprm-xuw7).

The fields are not the same in every dataset (estacio / estaci, valor / cabal_riu_m3_s,
with or without hour...): the schema is resolved from a sample of each dataset.

  uv run llobregat aca-opendata search
  uv run llobregat aca-opendata probe [--ids ...]
  uv run llobregat aca-opendata stations --dataset cabals_2010_2019
  uv run llobregat aca-opendata data --dataset cabals_2010_2019 \
      --stations-file data/raw/aca_opendata/stations_cabals_2010_2019.csv

Output in data/raw/aca_opendata/:
  stations_<dataset>.csv              available variables (column use)
  readings/dataset=<dataset>/station=<series>/year=<y>.parquet   (daily: all.parquet)
  manifest_<dataset>.csv              rows downloaded per series and year
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from llobregat.ingest.xema_opendata import SocrataClient, soql_ts, year_ranges

log = logging.getLogger("aca_opendata")


@dataclass(frozen=True)
class Spec:
    id: str
    first_year: int | None = None  # to sample variables, and default for --start
    last_year: int | None = None  # None = still being updated
    daily: bool = False


DATASETS = {
    "cabals": Spec("3yr3-vq6y", 2020, None),
    "cabals_2010_2019": Spec("9evs-b5nk", 2010, 2019),
    "cabals_1996_2009": Spec("mmzz-keb8", 1996, 2009),
    "embassaments": Spec("vjx7-6kcp", 2020, None),
    "cabals_diaris": Spec("2bwe-vgpr", 1951, None, daily=True),
    "embassaments_diaris": Spec("gn9e-3qhr", None, None, daily=True),
}

ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
STATION_HINTS = ("sensor", "codi", "component", "estaci", "punt", "id", "nom")
NUMERIC = re.compile(r"^-?\d+([.,]\d+)?$")

DATE_CANDIDATES = ("dia", "data", "fecha", "date")
META_CANDIDATES = [
    "conca",
    "subconca",
    "codi_estacio",
    "estacio",
    "estaci",
    "tipus_variable",
    "codi_variable",
    "descripcio_variable",
    "descripci_variable",
    "unitat_mesura",
]
SERIES_CANDIDATES = ("codi_variable", "codi_estacio", "estacio", "estaci")
NOT_VALUES = {"codi_variable", "utm_x", "utm_y", "lat", "long", "latitud", "longitud"}
# Words that set use=1 by default (searched in the type and the description)
DEFAULT_USE = {"cabals": ("cabal",), "embassaments": ("volum",)}


def resolve_spec(name: str) -> Spec:
    if name in DATASETS:
        return DATASETS[name]
    if re.fullmatch(r"[a-z0-9]{4}-[a-z0-9]{4}", name):
        return Spec(name)
    sys.exit(f"Unknown dataset: {name}. Options: {', '.join(DATASETS)} or an identifier")


def use_keys(name: str, match: list[str] | None) -> tuple[str, ...]:
    if match:
        return tuple(m.lower() for m in match)
    return DEFAULT_USE["embassaments" if "embassament" in name else "cabals"]


@dataclass
class Schema:
    date: str
    hour: str | None
    series: str
    meta: list[str]
    values: list[str]
    quality: str | None


def schema_from_rows(rows: list[dict]) -> Schema:
    det = detect_fields(rows)
    fields = set(det["fields"])
    date_f = next((f for f in DATE_CANDIDATES if f in fields), det["time"])
    if not date_f:
        sys.exit(f"Cannot find the date field among {sorted(fields)}")
    series = next((f for f in SERIES_CANDIDATES if f in fields), None)
    if not series:
        sys.exit(f"Cannot find a station or variable field among {sorted(fields)}")
    values = ["valor"] if "valor" in fields else [f for f in det["numeric"] if f not in NOT_VALUES]
    return Schema(
        date=date_f,
        hour="hora" if "hora" in fields else None,
        series=series,
        meta=[f for f in META_CANDIDATES if f in fields],
        values=values,
        quality="nivell_qualitat" if "nivell_qualitat" in fields else None,
    )


def resolve_schema(client: SocrataClient, ds: str) -> Schema:
    sc = schema_from_rows(client.get(ds, {"$limit": 200}))
    log.debug("Schema of %s: %s", ds, sc)
    return sc


def combine_time(dia: pd.Series, hora: pd.Series | None) -> pd.Series:
    """dia ('2026-11-20T00:00:00.000') + hora ('01:50') -> UTC timestamp."""
    d = dia.astype(str).str[:10]
    h = (
        hora.astype(str).str.strip().str[:5]
        if hora is not None
        else pd.Series("00:00", index=d.index)
    )
    ts = pd.to_datetime(d + " " + h, format="%Y-%m-%d %H:%M", errors="coerce")
    m24 = h.eq("24:00")  # '24:00' (if present) = 00:00 of the next day
    if m24.any():
        ts[m24] = pd.to_datetime(d[m24]) + pd.Timedelta(days=1)
    return ts.dt.tz_localize("UTC")


def month_ranges(a: date, b: date) -> list[tuple[date, date]]:
    """Split [a, b) into monthly chunks."""
    out, cur = [], a
    while cur < b:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        out.append((cur, min(nxt, b)))
        cur = nxt
    return out


def soql_like(field: str, text: str) -> str:
    esc = text.upper().replace("'", "''")
    return f"upper({field}) like '%{esc}%'"


def sample_windows(spec: Spec, args: argparse.Namespace) -> list[tuple[date, date] | None]:
    """Time windows to discover variables without grouping the whole dataset."""
    if args.date_from or args.date_to:
        a = date.fromisoformat(args.date_from) if args.date_from else date(1900, 1, 1)
        b = date.fromisoformat(args.date_to) if args.date_to else date.today() + timedelta(days=1)
        return [(a, b)]
    if spec.daily:
        return [None]  # small datasets: grouped whole
    if spec.last_year is None:
        return [(date.today() - timedelta(days=args.days), date.today() + timedelta(days=1))]
    # Closed historical datasets: January of each year
    return [(date(y, 1, 1), date(y, 2, 1)) for y in range(spec.first_year, spec.last_year + 1)]


# --------------------------------------------------------------------------- #
# Field detection
# --------------------------------------------------------------------------- #
def detect_fields(rows: list[dict]) -> dict:
    """Detect the time field, the station candidates and the numeric fields."""
    if not rows:
        return {"time": None, "station": [], "numeric": [], "fields": []}
    fields = sorted({k for r in rows for k in r if not k.startswith(":")})
    values = {f: [r.get(f) for r in rows if r.get(f) not in (None, "")] for f in fields}

    def all_match(f, pattern):
        v = values[f]
        return bool(v) and all(isinstance(x, str) and pattern.match(x) for x in v)

    time_fields = [f for f in fields if all_match(f, ISO_TS)]
    numeric = [f for f in fields if f not in time_fields and all_match(f, NUMERIC)]
    text = [
        f
        for f in fields
        if f not in time_fields and f not in numeric and all(isinstance(x, str) for x in values[f])
    ]
    station = sorted(text, key=lambda f: (not any(h in f.lower() for h in STATION_HINTS), f))
    # Prefer a time field with an hour (not just a date)
    time_fields.sort(key=lambda f: ("hora" not in f.lower() and "temps" not in f.lower(), f))
    return {
        "time": time_fields[0] if time_fields else None,
        "time_candidates": time_fields,
        "station": station,
        "numeric": numeric,
        "fields": fields,
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "sense_nom"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
CATALOG_URL = "https://analisi.transparenciacatalunya.cat/api/catalog/v1"
SEARCH_TERMS = ["cabal", "aforament", "embassament", "pluviòmetre", "Agència Catalana de l'Aigua"]


def cmd_search(client: SocrataClient, args: argparse.Namespace) -> None:
    """Search the portal catalogue for every dataset that may hold ACA data."""
    found: dict[str, dict] = {}
    for q in args.q or SEARCH_TERMS:
        resp = client.session.get(CATALOG_URL, params={"q": q, "limit": 100}, timeout=60)
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            res = r.get("resource", {})
            if res.get("type") not in (None, "dataset"):
                continue
            found[res["id"]] = {
                "id": res["id"],
                "nom": res.get("name"),
                "publicador": res.get("attribution"),
                "actualitzat": (res.get("data_updated_at") or res.get("updatedAt") or "")[:10],
                "camps": ", ".join(res.get("columns_field_name", [])[:15]),
            }
    df = pd.DataFrame(found.values())
    if df.empty:
        sys.exit("The catalogue returned no dataset")
    aca = df["publicador"].fillna("").str.contains("Aigua", case=False) | df["nom"].str.contains(
        "cabal|aforament|embassament|pluvi", case=False, regex=True
    )
    df = df[aca].sort_values("nom")
    path = Path(args.out) / "aca_catalog.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(df[["id", "nom", "publicador", "actualitzat"]].to_string(index=False))
    print(f"\n{len(df)} datasets saved to {path}. To see their period:")
    print("  uv run llobregat aca-opendata probe --ids " + " ".join(df["id"].head(5)))


def cmd_probe(client: SocrataClient, args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    targets = {i: i for i in args.ids} if args.ids else {k: s.id for k, s in DATASETS.items()}
    for name, ds in targets.items():
        print(f"\n=== {name} ({ds}) ===")
        try:
            rows = client.get(ds, {"$limit": 200})
        except RuntimeError as exc:
            print("Not available:", str(exc)[:200])
            continue
        det = detect_fields(rows)
        info = {"dataset": ds, "detected": det, "sample": rows[:3]}
        tf = next((f for f in DATE_CANDIDATES if f in det["fields"]), det["time"])
        if tf:
            try:
                agg = client.get(
                    ds,
                    {
                        "$select": f"min({tf}) as first_date, max({tf}) as last_date, "
                        "count(*) as n_rows"
                    },
                )
                info["range"] = agg[0] if agg else None
            except RuntimeError as exc:
                info["range"] = f"not available: {exc}"[:200]
        report[name] = info
        print("Fields:", det["fields"])
        print("Period and rows:", info.get("range"))
        print("Example:", json.dumps(rows[0] if rows else {}, ensure_ascii=False, indent=2))
    name = "probe_" + "_".join(args.ids) + ".json" if args.ids else "probe.json"
    (out / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out / name}")


def cmd_stations(client: SocrataClient, args: argparse.Namespace) -> None:
    """List the variables (or stations) of a dataset with their period and row counts.

    Grouping a whole 5-minute dataset (tens of millions of rows) exceeds the portal timeout:
    a recent window is sampled (live datasets) or January of each year (closed datasets).
    The daily ones, which are small, are grouped whole.
    """
    spec = resolve_spec(args.dataset)
    sc = resolve_schema(client, spec.id)
    group = ", ".join(sc.meta)
    filters = []
    filtered = bool(args.station_code or args.name)
    conca = args.basin if args.basin is not None else ([] if filtered else ["Llobregat"])
    if conca and "conca" in sc.meta:
        filters.append("(" + " OR ".join(soql_like("conca", c) for c in conca) + ")")
    if args.station_code and "codi_estacio" in sc.meta:
        codes = ", ".join("'" + c.replace("'", "''") + "'" for c in args.station_code)
        filters.append(f"codi_estacio in ({codes})")
    if args.name:
        name_f = next(
            (
                f
                for f in ("estacio", "estaci", "descripcio_variable", "descripci_variable")
                if f in sc.meta
            ),
            None,
        )
        if name_f:
            filters.append("(" + " OR ".join(soql_like(name_f, n) for n in args.name) + ")")

    parts = []
    windows = sample_windows(spec, args)
    for w in windows:
        where = list(filters)
        if w is not None:
            where.insert(0, f"{sc.date} >= '{soql_ts(w[0])}' AND {sc.date} < '{soql_ts(w[1])}'")
        params = {
            "$select": f"{group}, count(*) as n_rows, min({sc.date}) as first_date, "
            f"max({sc.date}) as last_date",
            "$group": group,
        }
        if where:
            params["$where"] = " AND ".join(where)
        log.info("%s: grouping %s", args.dataset, f"{w[0]} - {w[1]}" if w else "the whole dataset")
        parts.append(pd.DataFrame(client.get_all(spec.id, params)))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        rec = (
            client.get(spec.id, {"$select": "conca, count(*) as n_rows", "$group": "conca"})
            if "conca" in sc.meta and len(windows) == 1 and windows[0] is None
            else []
        )
        sys.exit(
            "No variable with these filters."
            + (f" Basins: {sorted(str(r.get('conca')) for r in rec)}" if rec else "")
        )
    for c in sc.meta:
        if c not in df.columns:
            df[c] = None
    df["n_rows"] = pd.to_numeric(df["n_rows"])
    df = (
        df.groupby(sc.meta, dropna=False)
        .agg(
            n_rows=("n_rows", "sum"),
            first_date=("first_date", "min"),
            last_date=("last_date", "max"),
        )
        .reset_index()
    )
    df["first_date"] = df["first_date"].astype(str).str[:10]
    df["last_date"] = df["last_date"].astype(str).str[:10]
    keys = use_keys(args.dataset, args.match)
    text_cols = [
        c
        for c in ("tipus_variable", "descripcio_variable", "descripci_variable")
        if c in df.columns
    ]
    text = (
        df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        if text_cols
        else pd.Series("", index=df.index)
    )
    df["use"] = text.apply(lambda x: int(any(k in x for k in keys))) if text_cols else 1
    df = df.sort_values([c for c in ("subconca", "codi_estacio", sc.series) if c in df.columns])
    tags = [*args.station_code, *args.name]
    suffix = ("_" + "-".join(safe_name(c) for c in tags)) if tags else ""
    path = Path(args.out) / f"stations_{safe_name(args.dataset)}{suffix}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # -sig: Excel reads it correctly
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 55)
    show = [
        c
        for c in (
            "subconca",
            "codi_estacio",
            "estacio",
            "estaci",
            "tipus_variable",
            "descripci_variable",
            "codi_variable",
            "unitat_mesura",
        )
        if c in df.columns
    ]
    print(df[[*show, "n_rows", "first_date", "last_date", "use"]].to_string(index=False))
    sampled = " (rows of the sample)" if len(windows) > 1 else ""
    print(f"\n{len(df)} series{sampled}, {int(df['use'].sum())} with use=1, saved to {path}")
    print(f"Series field: {sc.series} · values: {sc.values}")


def _frame(raw: pd.DataFrame, sc: Schema) -> pd.DataFrame:
    out = pd.DataFrame(
        {"timestamp": combine_time(raw[sc.date], raw.get(sc.hour) if sc.hour else None)}
    )
    out[sc.series] = raw[sc.series].astype("string")
    if "codi_estacio" in raw.columns and sc.series != "codi_estacio":
        out["codi_estacio"] = raw["codi_estacio"].astype("string")
    vals = [v for v in sc.values if v in raw.columns]
    if len(vals) == 1:
        out["value"] = pd.to_numeric(raw[vals[0]], errors="coerce")
    else:
        for v in vals:
            out[v] = pd.to_numeric(raw[v], errors="coerce")
    if sc.quality and sc.quality in raw.columns:
        out["nivell_qualitat"] = raw[sc.quality].astype("string")
    n = len(out)
    out = out.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    if n > 1.2 * len(out):
        log.debug("%d duplicate rows removed", n - len(out))
    return out.sort_values("timestamp")


def cmd_data(client: SocrataClient, args: argparse.Namespace) -> None:
    spec = resolve_spec(args.dataset)
    sc = resolve_schema(client, spec.id)
    st = pd.read_csv(args.stations_file, dtype=str, encoding="utf-8-sig")
    if "use" in st.columns:
        st = st[st["use"].astype(str).str.strip().isin(["1", "true", "True", "sí", "si"])]
    if sc.series not in st.columns:
        sys.exit(f"The file {args.stations_file} has no column {sc.series}")
    series = sorted(st[sc.series].dropna().unique())
    default_start = date(spec.first_year or 1950, 1, 1)
    default_end = date(spec.last_year + 1, 1, 1) if spec.last_year else date.today() + timedelta(1)
    start = date.fromisoformat(args.start) if args.start else default_start
    end = date.fromisoformat(args.end) if args.end else default_end
    out = Path(args.out) / "readings" / f"dataset={safe_name(args.dataset)}"
    select = ", ".join(
        dict.fromkeys(
            [
                sc.date,
                *([sc.hour] if sc.hour else []),
                sc.series,
                *(["codi_estacio"] if "codi_estacio" in sc.meta else []),
                *sc.values,
                *([sc.quality] if sc.quality else []),
            ]
        )
    )
    log.info("%d series of %s, %s - %s", len(series), args.dataset, start, end)

    def fetch(var: str, a: date, b: date) -> list[dict]:
        esc = str(var).replace("'", "''")
        return client.get_all(
            spec.id,
            {
                "$select": select,
                "$where": f"{sc.series} = '{esc}' AND {sc.date} >= '{soql_ts(a)}' "
                f"AND {sc.date} < '{soql_ts(b)}'",
                "$order": ", ".join([sc.date, *([sc.hour] if sc.hour else [])]),
            },
            page_size=args.page_size,
        )

    manifest = []
    for k, var in enumerate(series, 1):
        vdir = out / f"station={safe_name(var)}"
        if spec.daily:
            path = vdir / "all.parquet"
            if path.exists() and not args.overwrite:
                manifest.append((var, None, len(pd.read_parquet(path)), "existent"))
                continue
            rows = fetch(var, start, end)
            if rows:
                vdir.mkdir(parents=True, exist_ok=True)
                df = _frame(pd.DataFrame(rows), sc)
                df.to_parquet(path, index=False)
                first = df["timestamp"].iloc[0].year
                manifest.append((var, first, len(df), "downloaded"))
                log.info("[%d/%d] %s: %d days since %d", k, len(series), var, len(df), first)
            else:
                manifest.append((var, None, 0, "buit"))
            continue
        for year, a, b in year_ranges(start, end):
            path = vdir / f"year={year}.parquet"
            empty = path.with_suffix(".empty")
            current = b >= date.today()
            if not args.overwrite and not current and (path.exists() or empty.exists()):
                n = len(pd.read_parquet(path)) if path.exists() else 0
                manifest.append((var, year, n, "existent"))
                continue
            rows = []
            for ma, mb in month_ranges(a, b):
                rows += fetch(var, ma, mb)
            vdir.mkdir(parents=True, exist_ok=True)
            if not rows:
                empty.touch()
                manifest.append((var, year, 0, "buit"))
                continue
            df = _frame(pd.DataFrame(rows), sc)
            df.to_parquet(path, index=False)
            manifest.append((var, year, len(df), "downloaded"))
            dup = f" ({len(rows) - len(df)} repetides)" if len(rows) > 1.2 * len(df) else ""
            log.info("[%d/%d] %s %d: %d rows%s", k, len(series), var, year, len(df), dup)
    m = pd.DataFrame(manifest, columns=[sc.series, "year", "rows", "status"])
    mpath = Path(args.out) / f"manifest_{safe_name(args.dataset)}.csv"
    m.to_csv(mpath, index=False)
    summary = (
        m[m["rows"] > 0]
        .groupby(sc.series)
        .agg(years=("year", "nunique"), first_year=("year", "min"), rows=("rows", "sum"))
    )
    print(summary.to_string())
    print(f"\nManifest in {mpath}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llobregat aca-opendata",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="data/raw/aca_opendata")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--timeout", type=int, default=300, help="Timeout per query (s)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe", help="Fields, sample and period of each dataset")
    pr.add_argument("--ids", nargs="*", default=[], help="Other datasets (e.g. hprm-xuw7)")
    se = sub.add_parser("search", help="Search the portal catalogue for ACA datasets")
    se.add_argument("--q", nargs="*", default=None, help="Search terms")

    help_ds = "Dataset: " + ", ".join(DATASETS) + " (or an identifier xxxx-xxxx)"
    st = sub.add_parser("stations", help="List of series with row counts and period")
    st.add_argument("--dataset", required=True, help=help_ds)
    st.add_argument(
        "--basin",
        nargs="*",
        default=None,
        help="Basins (text within the conca field). Llobregat by default if there is no "
        "other filter; --basin without values = all",
    )
    st.add_argument(
        "--station-code",
        nargs="*",
        default=[],
        help="Only these stations (e.g. E12 E13 E14); saved to a separate CSV",
    )
    st.add_argument(
        "--name",
        nargs="*",
        default=[],
        help='Text within the station name (e.g. Baells Llosa "Sant Pon")',
    )
    st.add_argument(
        "--days",
        type=int,
        default=14,
        help="Live datasets: recent window to discover variables (days)",
    )
    st.add_argument("--from", dest="date_from", default=None, help="Custom window: start")
    st.add_argument("--to", dest="date_to", default=None, help="Custom window: end (excluded)")
    st.add_argument(
        "--match",
        nargs="*",
        default=None,
        help="Words of the type or description that set use=1 (default: cabal or volum)",
    )
    d = sub.add_parser("data", help="Download per series and year (can be resumed)")
    d.add_argument("--dataset", required=True, help=help_ds)
    d.add_argument("--stations-file", required=True)
    d.add_argument("--start", default=None, help="By default, the start of the dataset")
    d.add_argument("--end", default=None)
    d.add_argument("--page-size", type=int, default=50_000)
    d.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    client = SocrataClient(app_token=os.environ.get("SOCRATA_APP_TOKEN"))
    client.timeout = args.timeout
    {"probe": cmd_probe, "search": cmd_search, "stations": cmd_stations, "data": cmd_data}[
        args.cmd
    ](client, args)

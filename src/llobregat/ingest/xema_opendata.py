#!/usr/bin/env python3
"""
Download of XEMA (Meteocat) data from the Catalan government open data portal
(Socrata / SODA API).

Datasets:
  nzvn-apee  XEMA weather data (half-hourly readings, since 2009)
  yqwd-vj5e  Metadata of the automatic weather stations
  4fb2-n3yi  Metadata of the weather variables

Recommended workflow:
  1. python download_xema.py probe
       Show the actual fields of each dataset (initial check).
  2. python download_xema.py metadata
       Save the station and variable metadata.
  3. python download_xema.py stations [--basin-geojson basin.geojson | --bbox ...]
       Build the list of candidate stations in the basin, to be reviewed by hand.
  4. python download_xema.py data --stations-file data/raw/xema/stations_candidates.csv
       Download the readings per station, variable and year (can be resumed).

Output (by default in data/raw/xema/):
  metadata/stations.parquet, metadata/variables.parquet (+ .csv)
  stations_candidates.csv
  readings/variable=<code>/station=<code>/year=<year>.parquet
  manifest.csv  (rows downloaded and expected, and completeness per station-year)

Dependencies: requests, pandas, pyarrow. Optional: shapely (for --basin-geojson).
Optional: environment variable SOCRATA_APP_TOKEN to reduce the risk of rate limiting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://analisi.transparenciacatalunya.cat/resource"
DATASETS = {
    "readings": "nzvn-apee",
    "stations": "yqwd-vj5e",
    "variables": "4fb2-n3yi",
}

# Approximate rectangle containing the Llobregat basin (lon_min, lat_min, lon_max, lat_max).
# It is deliberately wide: it includes stations outside the basin, to be reviewed by hand.
# For an exact selection, use --basin-geojson with the basin polygon.
LLOBREGAT_BBOX = (1.30, 41.25, 2.25, 42.40)

DEFAULT_VARIABLES = ["35"]  # 35 = precipitation (mm), according to Meteocat
SLOTS_PER_DAY = 48  # half-hourly readings

log = logging.getLogger("xema")


# --------------------------------------------------------------------------- #
# Socrata client
# --------------------------------------------------------------------------- #
@dataclass
class SocrataClient:
    app_token: str | None = None
    timeout: int = 120
    max_retries: int = 6
    pause: float = 0.3  # pause between requests, to be gentle with the server

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        if self.app_token:
            self.session.headers["X-App-Token"] = self.app_token

    def get(self, dataset: str, params: dict) -> list[dict]:
        url = f"{BASE_URL}/{dataset}.json"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                wait = min(2**attempt, 60)
                log.warning("Network error (%s). Retry %d in %ds", exc, attempt, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                time.sleep(self.pause)
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2**attempt, 60)
                log.warning("HTTP %s. Retry %d in %ds", resp.status_code, attempt, wait)
                time.sleep(wait)
                continue
            # 4xx errors that retrying will not fix (malformed query, missing field...)
            raise RuntimeError(f"HTTP {resp.status_code} at {resp.url}\n{resp.text[:500]}")
        raise RuntimeError(f"Retries exhausted for {url} with {params}")

    def get_all(self, dataset: str, params: dict, page_size: int = 50_000) -> list[dict]:
        """Download every row, paging with $limit/$offset."""
        rows: list[dict] = []
        offset = 0
        while True:
            page = self.get(dataset, {**params, "$limit": page_size, "$offset": offset})
            rows.extend(page)
            if len(page) < page_size:
                return rows
            offset += page_size


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Find a column by exact name or by partial match (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for low, orig in lower.items():
            if cand.lower() in low:
                return orig
    return None


def soql_ts(d: date | datetime) -> str:
    """Date format for SoQL (floating timestamp)."""
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{d.isoformat()}T00:00:00"


def write_table(df: pd.DataFrame, path: Path, csv: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if csv:
        df.to_csv(path.with_suffix(".csv"), index=False)


def year_ranges(start: date, end: date) -> list[tuple[int, date, date]]:
    """Split [start, end) into calendar-year chunks."""
    out = []
    y = start.year
    while date(y, 1, 1) < end:
        a = max(start, date(y, 1, 1))
        b = min(end, date(y + 1, 1, 1))
        if a < b:
            out.append((y, a, b))
        y += 1
    return out


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_probe(client: SocrataClient, args: argparse.Namespace) -> None:
    for name, ds in DATASETS.items():
        rows = client.get(ds, {"$limit": 3})
        print(f"\n=== {name} ({ds}) ===")
        if not rows:
            print("(no rows)")
            continue
        print("Fields:", sorted({k for r in rows for k in r}))
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))


def cmd_metadata(client: SocrataClient, args: argparse.Namespace) -> None:
    out = Path(args.out) / "metadata"
    for name in ("stations", "variables"):
        df = pd.DataFrame(client.get_all(DATASETS[name], {}))
        write_table(df, out / f"{name}.parquet", csv=True)
        log.info("%s: %d rows saved to %s", name, len(df), out)


def load_stations_metadata(client: SocrataClient, out: Path) -> pd.DataFrame:
    path = out / "metadata" / "stations.parquet"
    if not path.exists():
        log.info("No station metadata; downloading it")
        df = pd.DataFrame(client.get_all(DATASETS["stations"], {}))
        write_table(df, path, csv=True)
    return pd.read_parquet(path)


def cmd_stations(client: SocrataClient, args: argparse.Namespace) -> None:
    out = Path(args.out)
    st = load_stations_metadata(client, out)

    lat_col = find_column(st, "latitud", "lat")
    lon_col = find_column(st, "longitud", "lon")
    code_col = find_column(st, "codi_estacio")
    if not (lat_col and lon_col and code_col):
        sys.exit(f"Cannot find the code/latitude/longitude columns. Columns: {list(st.columns)}")

    st = st.copy()
    st["lat"] = pd.to_numeric(st[lat_col], errors="coerce")
    st["lon"] = pd.to_numeric(st[lon_col], errors="coerce")

    if args.basin_geojson:
        try:
            from shapely.geometry import Point, shape
            from shapely.ops import unary_union
        except ImportError:
            sys.exit("--basin-geojson needs shapely installed")
        gj = json.loads(Path(args.basin_geojson).read_text(encoding="utf-8"))
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        basin = unary_union([shape(f.get("geometry", f)) for f in feats])
        if args.buffer_km:
            basin = basin.buffer(args.buffer_km / 111.0)  # approximation: degrees ≈ km/111
        st["in_basin"] = [
            bool(pd.notna(lo) and pd.notna(la) and basin.contains(Point(lo, la)))
            for lo, la in zip(st["lon"], st["lat"], strict=False)
        ]
        method = f"polygon {args.basin_geojson}"
    else:
        lon_min, lat_min, lon_max, lat_max = args.bbox
        st["in_basin"] = st["lon"].between(lon_min, lon_max) & st["lat"].between(lat_min, lat_max)
        method = f"rectangle {args.bbox} (approximate: review the list)"

    cand = st[st["in_basin"]].copy()
    cand["use"] = 1  # set it to 0 by hand to exclude a station
    keep = (
        [code_col]
        + [
            c
            for c in (
                find_column(st, "nom_estacio"),
                find_column(st, "nom_municipi"),
                find_column(st, "nom_comarca"),
                find_column(st, "altitud"),
                find_column(st, "nom_estat_ema", "estat"),
                find_column(st, "data_inici"),
                find_column(st, "data_fi"),
            )
            if c
        ]
        + ["lat", "lon", "use"]
    )
    cand = cand[list(dict.fromkeys(keep))].rename(columns={code_col: "codi_estacio"})
    cand = cand.sort_values(["lat", "lon"], ascending=[False, True])

    path = out / "stations_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(path, index=False)
    log.info("Selection by %s", method)
    log.info("%d candidate stations saved to %s", len(cand), path)
    print(cand.to_string(index=False))


def read_station_list(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype=str)
    col = find_column(df, "codi_estacio")
    if col is None:
        sys.exit(f"{path} has no codi_estacio column")
    if "use" in df.columns:
        df = df[df["use"].astype(str).str.strip().isin(["1", "true", "True", "sí", "si"])]
    return sorted(df[col].dropna().str.strip().unique())


def normalise_readings(rows: list[dict], assume_utc: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ts_col = find_column(df, "data_lectura")
    val_col = find_column(df, "valor_lectura")
    if ts_col is None or val_col is None:
        raise RuntimeError(f"Missing data_lectura/valor_lectura. Columns: {list(df.columns)}")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if assume_utc:
        df[ts_col] = df[ts_col].dt.tz_localize("UTC")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    for c in df.columns:
        if c not in (ts_col, val_col):
            df[c] = df[c].astype("string")
    df = df.sort_values(ts_col).drop_duplicates()
    return df


def cmd_data(client: SocrataClient, args: argparse.Namespace) -> None:
    out = Path(args.out)
    stations = args.stations or read_station_list(Path(args.stations_file))
    if not stations:
        sys.exit("No stations to download")
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    chunks = year_ranges(start, end)

    log.info(
        "%d stations × %d variables × %d years = %d chunks",
        len(stations),
        len(args.variables),
        len(chunks),
        len(stations) * len(args.variables) * len(chunks),
    )

    manifest_rows = []
    for var in args.variables:
        for stn in stations:
            for year, a, b in chunks:
                path = (
                    out / "readings" / f"variable={var}" / f"station={stn}" / f"year={year}.parquet"
                )
                empty_marker = path.with_suffix(".empty")
                expected = (b - a).days * SLOTS_PER_DAY
                is_current = b >= date.today() - timedelta(days=1)

                # Resume: chunks already downloaded are skipped, except the current year
                if not args.overwrite and not is_current:
                    if path.exists():
                        n = len(pd.read_parquet(path, columns=None))
                        manifest_rows.append((stn, var, year, n, expected, "existing"))
                        continue
                    if empty_marker.exists():
                        manifest_rows.append((stn, var, year, 0, expected, "empty"))
                        continue

                params = {
                    "codi_estacio": stn,
                    "codi_variable": var,
                    "$where": f"data_lectura >= '{soql_ts(a)}' AND data_lectura < '{soql_ts(b)}'",
                    "$order": "data_lectura",
                }
                rows = client.get_all(DATASETS["readings"], params, page_size=args.page_size)
                df = normalise_readings(rows, assume_utc=not args.no_utc)
                if df.empty:
                    empty_marker.parent.mkdir(parents=True, exist_ok=True)
                    empty_marker.touch()
                    manifest_rows.append((stn, var, year, 0, expected, "empty"))
                    log.info("%s var %s %d: no data", stn, var, year)
                    continue
                write_table(df, path)
                manifest_rows.append((stn, var, year, len(df), expected, "downloaded"))
                log.info(
                    "%s var %s %d: %d rows (%.1f%% of the maximum)",
                    stn,
                    var,
                    year,
                    len(df),
                    100 * len(df) / expected,
                )

    manifest = pd.DataFrame(
        manifest_rows,
        columns=["codi_estacio", "codi_variable", "year", "rows", "rows_expected", "status"],
    )
    manifest["completeness"] = (manifest["rows"] / manifest["rows_expected"]).round(4)
    manifest.to_csv(out / "manifest.csv", index=False)
    log.info("Manifest saved to %s", out / "manifest.csv")

    summary = (
        manifest[manifest["rows"] > 0]
        .groupby(["codi_variable", "codi_estacio"])
        .agg(
            years=("year", "nunique"),
            first_year=("year", "min"),
            completeness=("completeness", "mean"),
        )
        .round(3)
    )
    print("\nSummary per station (mean completeness of the years with data):")
    print(summary.to_string())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="data/raw/xema", help="Output folder")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="Show the actual fields of each dataset")
    sub.add_parser("metadata", help="Download station and variable metadata")

    s = sub.add_parser("stations", help="List of candidate stations in the basin")
    s.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=list(LLOBREGAT_BBOX),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    s.add_argument("--basin-geojson", help="Basin polygon (GeoJSON, WGS84)")
    s.add_argument(
        "--buffer-km",
        type=float,
        default=0.0,
        help="Margin around the polygon, in km (includes nearby stations)",
    )

    d = sub.add_parser("data", help="Download readings per station, variable and year")
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--stations-file", help="CSV with a codi_estacio column (and optionally use)")
    g.add_argument("--stations", nargs="+", help="Station codes separated by spaces")
    d.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="Variable codes (default 35, precipitation)",
    )
    d.add_argument("--start", default="2009-01-01", help="Start date (YYYY-MM-DD)")
    d.add_argument("--end", default=None, help="End date, excluded (default: today)")
    d.add_argument("--page-size", type=int, default=50_000)
    d.add_argument("--overwrite", action="store_true", help="Download everything again")
    d.add_argument(
        "--no-utc",
        action="store_true",
        help="Do not mark the dates as UTC (UTC is assumed by default; to be verified)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    client = SocrataClient(app_token=os.environ.get("SOCRATA_APP_TOKEN"))
    {"probe": cmd_probe, "metadata": cmd_metadata, "stations": cmd_stations, "data": cmd_data}[
        args.cmd
    ](client, args)


if __name__ == "__main__":
    main()

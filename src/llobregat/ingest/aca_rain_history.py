"""Reader for the ACA rain gauge history (one Excel file per basin, supplied on request).

Format of the files supplied by the ACA (Precipitació_<Basin>.xlsx):
  - One or more sheets (the series does not fit in Excel's row limit); they are concatenated.
  - Row 2: station code and name ("EA015_Sant Celoni"); row 3: variable; row 4: unit (mm/h).
  - Column B: date and time (UTC+0, according to the ACA); columns C...: one per gauge.
  - Value: mean intensity of the 5-minute interval in mm/h. Rain (mm) = value / 12.
    Values are multiples of 1.2 mm/h (0.1 mm bucket tips).
  - Until 2010, event-based logging: there are rows only when it rains, plus a zero roughly
    every hour. Since 2010, one row every 5 minutes.

Treatment:
  - Event-based period: an interval without a row counts as 0 mm if the station has a
    reading in the previous HEARTBEAT minutes; otherwise it counts as missing.
  - Regular period: an empty cell is missing data (it is not turned into 0).
  - Conservative time convention: the label is treated as the start of the interval and
    shifted +5 min (end of the interval), as with XEMA.
These are raw, unvalidated data (ACA notice): they need quality control.

  uv run llobregat aca-rain --input data/raw/aca_rain --basins Llobregat
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("aca_rain")

STEP = pd.Timedelta(minutes=5)
SLOTS_PER_MONTH_DENSE = 0.9  # fraction of the grid from which a month counts as "regular"
HEARTBEAT_MIN = 75


def parse_station(header: str) -> tuple[str, str]:
    """'EA015_Sant Celoni' -> ('EA015', 'Sant Celoni')."""
    m = re.match(r"^\s*([A-Za-z0-9]+)_(.*)$", str(header))
    return (m.group(1), m.group(2).strip()) if m else (str(header).strip(), str(header).strip())


def _time_column(raw: pd.DataFrame) -> int:
    """Position of the date-time column (the first one with a majority of dates)."""
    for j in range(raw.shape[1]):
        col = raw.iloc[:, j]
        if pd.to_datetime(col, errors="coerce").notna().mean() > 0.5:
            return j
    raise ValueError("no date-time column found")


def read_workbook(path: str | Path) -> pd.DataFrame:
    """Read every sheet and return a wide table: t + one column per station (mm/h)."""
    sheets = pd.read_excel(path, sheet_name=None, engine="calamine", header=None)
    frames, names = [], None
    for i, (_, raw) in enumerate(sheets.items()):
        if raw.dropna(how="all").empty:
            continue
        jt = _time_column(raw)
        ts = pd.to_datetime(raw.iloc[:, jt], errors="coerce")
        first = int(np.argmax(ts.notna().to_numpy()))
        if names is None:
            if first == 0:
                raise ValueError(f"{path}: the first sheet has no header")
            head = raw.iloc[:first, jt + 1 :].dropna(axis=1, how="all")
            names = [str(v) for v in head.bfill().iloc[0]]
            cols = list(head.columns)
        body = raw.iloc[first:].reindex(columns=[raw.columns[jt], *cols])
        body.columns = ["t", *names]
        frames.append(body)
        log.info("%s, fulla %d: %d files", Path(path).name, i + 1, len(body))
    df = pd.concat(frames, ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], errors="coerce").dt.round("min")
    df = df.dropna(subset=["t"]).drop_duplicates("t", keep="last").sort_values("t")
    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def dense_since(times: pd.Series) -> pd.Timestamp | None:
    """First month from which the series has a row for almost every 5-min interval."""
    per_month = times.dt.to_period("M").value_counts().sort_index()
    slots = per_month.index.to_timestamp().days_in_month * 288
    dense = (per_month.to_numpy() / slots.to_numpy()) >= SLOTS_PER_MONTH_DENSE
    # first month from which all months are regular (tolerating isolated gaps)
    for k in range(len(dense)):
        if dense[k] and dense[k:].mean() >= 0.95:
            return per_month.index[k].to_timestamp()
    return None


def to_regular(
    df: pd.DataFrame, heartbeat_min: int = HEARTBEAT_MIN
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Regular 5-min grid with the implicit-zero rule in the event-based period."""
    cut = dense_since(df["t"])
    grid = pd.date_range(df["t"].min(), df["t"].max(), freq=STEP)
    wide = df.set_index("t").reindex(grid)
    sparse = grid < cut if cut is not None else np.ones(len(grid), dtype=bool)
    hb = int(heartbeat_min / 5)
    for c in wide.columns:
        s = wide[c]
        seen = s.notna().astype(float).where(s.notna())  # 1 where there is a reading
        recent = seen.ffill(limit=hb).notna()
        fill0 = sparse & s.isna() & recent.to_numpy()
        wide[c] = s.mask(fill0, 0.0)
    wide.index.name = "t"
    return wide, cut


def to_long(wide: pd.DataFrame, basin: str, column: str, shift_to_end: bool = True) -> pd.DataFrame:
    """Common long format (one row per interval) for one station, with rain in mm."""
    s = wide[column].dropna()
    code, _ = parse_station(column)
    t = s.index + (STEP if shift_to_end else pd.Timedelta(0))
    n = len(s)
    return pd.DataFrame(
        {
            "source": pd.Categorical(["aca_rain"] * n),
            "station": pd.Categorical([code] * n),
            "variable": pd.Categorical(["rain_mm"] * n),
            "timestamp": t.tz_localize("UTC"),
            "value": (s.to_numpy() / 12.0).round(4),
            "flag": pd.Categorical([basin] * n),
        }
    )


def station_table(wide: pd.DataFrame, basin: str, cut) -> pd.DataFrame:
    rows = []
    for h in wide.columns:
        code, name = parse_station(h)
        s = wide[h]
        valid = s.notna()
        if not valid.any():
            continue
        yearly = (s / 12).groupby(s.index.year).sum(min_count=1)
        cov_reg = float(valid[s.index >= cut].mean()) if cut is not None else np.nan
        rows.append(
            {
                "code": code,
                "name": name,
                "basin": basin,
                "start": s.index[valid.to_numpy()][0],
                "end": s.index[valid.to_numpy()][-1],
                "coverage": round(float(valid.mean()), 3),
                "coverage_since_regular": round(cov_reg, 3),
                "median_annual_rain_mm": round(float(yearly[yearly > 0].median()), 0),
            }
        )
    return pd.DataFrame(rows)


def process_file(path: Path, out_dir: Path) -> pd.DataFrame:
    """Process a basin Excel file and save one Parquet per gauge (bounded memory)."""
    basin = re.sub(r"^Precipitaci[oó]_", "", path.stem)
    wide, cut = to_regular(read_workbook(path))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for col in wide.columns:
        long = to_long(wide, basin, col)
        if long.empty:
            continue
        long.to_parquet(out_dir / f"{parse_station(col)[0]}.parquet", index=False)
        n += len(long)
    st = station_table(wide, basin, cut)
    log.info(
        "%s: %d gauges, %d readings, regular logging since %s",
        basin,
        len(st),
        n,
        cut.date() if cut is not None else "mai",
    )
    return st


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat aca-rain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", default="data/raw/aca_rain")
    p.add_argument("--out", default="data/interim/aca_rain")
    p.add_argument(
        "--basins",
        nargs="*",
        default=None,
        help="Basins to process (e.g. Llobregat Foix); all by default",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    files = sorted(Path(args.input).glob("Precipitaci*.xlsx"))
    if args.basins:
        wanted = {b.lower() for b in args.basins}
        files = [f for f in files if re.sub(r"^Precipitaci[oó]_", "", f.stem).lower() in wanted]
    if not files:
        raise SystemExit(f"No Precipitació_*.xlsx files in {args.input}")
    tables = [process_file(f, Path(args.out)) for f in files]
    st = pd.concat(tables, ignore_index=True)
    st.to_csv(Path(args.out) / "stations.csv", index=False)
    pd.set_option("display.width", 200)
    print(st.to_string(index=False))

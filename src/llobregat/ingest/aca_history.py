"""Generic reader for ACA history files (CSV or Excel) with a configurable column mapping.

The reader is configured with a column mapping, so a new file layout only needs a new
mapping.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class HistoryMapping(BaseModel):
    timestamp_col: str
    value_col: str
    station_col: str | None = None  # if the file has several stations
    station: str | None = None  # if the file has a single station
    variable: str = "flow"
    timezone: str = "UTC"  # e.g. "Europe/Madrid" if the ACA supplies local time
    datetime_format: str | None = None
    decimal: str = "."
    sep: str = ","
    sheet_name: str | int = 0
    skiprows: int = 0
    flag_col: str | None = None


def to_utc(ts: pd.Series, timezone: str) -> pd.Series:
    """Convert local times to UTC handling the clock changes.

    Repeated hours in October are set to NaT (it cannot be known which one is meant), and
    non-existent hours in March are shifted forward.
    """
    if timezone.upper() == "UTC":
        return ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    local = ts.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC")


def load_aca_history(path: str | Path, mapping: HistoryMapping) -> pd.DataFrame:
    """Read an ACA file (CSV or Excel) and convert it to the common long format."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, sheet_name=mapping.sheet_name, skiprows=mapping.skiprows)
    else:
        raw = pd.read_csv(path, sep=mapping.sep, decimal=mapping.decimal, skiprows=mapping.skiprows)
    if mapping.station_col is None and mapping.station is None:
        raise ValueError("The mapping needs station_col or station")

    ts = pd.to_datetime(raw[mapping.timestamp_col], format=mapping.datetime_format)
    out = pd.DataFrame(
        {
            "source": "aca",
            "station": raw[mapping.station_col].astype(str)
            if mapping.station_col
            else mapping.station,
            "variable": mapping.variable,
            "timestamp": to_utc(ts, mapping.timezone),
            "value": pd.to_numeric(raw[mapping.value_col], errors="coerce"),
            "flag": raw[mapping.flag_col].astype(str) if mapping.flag_col else None,
        }
    )
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)

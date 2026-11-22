"""Parquet storage partitioned by source, year and month, with deduplication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llobregat.ingest import OBS_COLUMNS

KEYS = ["source", "station", "variable", "timestamp"]


def validate_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Check and normalise the common long format."""
    missing = set(OBS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df[OBS_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    for c in ("source", "station", "variable", "flag"):
        out[c] = out[c].astype("string")
    return out


def month_path(root: Path, source: str, ts: pd.Timestamp) -> Path:
    return root / source / f"year={ts.year}" / f"month={ts.month:02d}.parquet"


def append_dedup(df: pd.DataFrame, root: str | Path) -> int:
    """Append observations to the store, one file per source and month, without duplicates.

    If an observation already exists (same source, station, variable and time), the most
    recently received one is kept: real-time data can be revised later.
    Returns the number of new rows.
    """
    if df.empty:
        return 0
    df = validate_obs(df)
    root = Path(root)
    n_new = 0
    month_key = df["timestamp"].dt.strftime("%Y-%m")
    for (source, _), part in df.groupby([df["source"], month_key]):
        path = month_path(root, str(source), part["timestamp"].iloc[0])
        if path.exists():
            old = pd.read_parquet(path)
            before = len(old)
            merged = pd.concat([old, part], ignore_index=True)
        else:
            before = 0
            merged = part
        merged = (
            merged.drop_duplicates(subset=KEYS, keep="last")
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        n_new += len(merged) - before
    return n_new


def read_store(
    root: str | Path,
    source: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read the store (optionally filtered by source and period)."""
    root = Path(root)
    pattern = f"{source}/**/*.parquet" if source else "**/*.parquet"
    files = sorted(root.glob(pattern))
    if not files:
        return pd.DataFrame(columns=OBS_COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def last_timestamp(root: str | Path, source: str, station: str, variable: str):
    """Last stored time for a series (or None)."""
    root = Path(root)
    for path in sorted((root / source).glob("year=*/month=*.parquet"), reverse=True):
        df = pd.read_parquet(path, columns=["station", "variable", "timestamp"])
        sel = df[(df["station"] == station) & (df["variable"] == variable)]
        if not sel.empty:
            return sel["timestamp"].max()
    return None

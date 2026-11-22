"""Reader for the XEMA history downloaded from the open data portal.

Converts the files in data/raw/xema/readings/ to the common long format. Before mid-2014,
some stations only have hourly readings (codi_base = HO); depending on the chosen policy
they are split into two half hours ("split") or discarded ("drop").

Time convention: data_lectura is treated as the start of the period (see
llobregat.resample.xema_period_start_to_end), also for the hourly readings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llobregat.ingest import OBS_COLUMNS


def split_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Split each hourly reading (HO) into two half-hourly readings of half the value."""
    hourly = df["codi_base"] == "HO"
    if not hourly.any():
        return df
    ho = df[hourly]
    first = ho.assign(valor_lectura=ho["valor_lectura"] / 2, codi_base="HO_SPLIT")
    second = first.assign(data_lectura=first["data_lectura"] + pd.Timedelta(minutes=30))
    return pd.concat([df[~hourly], first, second], ignore_index=True).sort_values("data_lectura")


def load_xema_history(
    root: str | Path,
    stations: list[str],
    variable: str = "35",
    hourly_policy: str = "split",
) -> pd.DataFrame:
    root = Path(root) / "readings" / f"variable={variable}"
    frames = []
    for st in stations:
        files = sorted((root / f"station={st}").glob("year=*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if hourly_policy == "split":
            df = split_hourly(df)
        elif hourly_policy == "drop":
            df = df[df["codi_base"] != "HO"]
        else:
            raise ValueError(f"Unknown hourly policy: {hourly_policy}")
        frames.append(
            pd.DataFrame(
                {
                    "source": "xema",
                    "station": st,
                    "variable": variable,
                    "timestamp": pd.to_datetime(df["data_lectura"], utc=True),
                    "value": pd.to_numeric(df["valor_lectura"], errors="coerce"),
                    "flag": df["codi_estat"].astype(str) + "|" + df["codi_base"].astype(str),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=OBS_COLUMNS)
    return pd.concat(frames, ignore_index=True)

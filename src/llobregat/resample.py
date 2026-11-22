"""From long-format observations to a regular 30-minute grid (UTC).

Convention: each value is labelled with the END of the interval it summarises, so the
row for time t only contains information observed up to t.
"""

from __future__ import annotations

import pandas as pd

FREQ = "30min"

# How each variable is aggregated within the interval
AGGREGATIONS: dict[str, list[str]] = {
    "flow": ["mean", "max"],
    "outflow": ["mean"],
    "volume": ["last"],
    "level": ["mean", "max"],
    "canal_flow": ["mean"],
    # ACA rain intensity (mm/h): the mean over the 30-min interval,
    # multiplied by 0.5 h, gives the rainfall in mm (see rain_intensity_to_mm)
    "rain_intensity": ["mean"],
    "35": ["sum"],  # XEMA precipitation (mm)
    "precip": ["sum"],
    "rain_mm": ["sum"],  # ACA rain gauges (history, 5 min)
}


def column_name(source: str, station: str, variable: str, agg: str | None = None) -> str:
    base = f"{source}:{station}:{variable}"
    return f"{base}:{agg}" if agg else base


def to_grid(s: pd.Series, how: str, freq: str = FREQ) -> pd.Series:
    """Aggregate a series to intervals (t - freq, t], labelled with t."""
    r = s.resample(freq, label="right", closed="right")
    if how == "sum":
        return r.sum(min_count=1)
    return getattr(r, how)()


def xema_period_start_to_end(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """Convert period-start labels to period-end labels.

    Applied by default to XEMA: if the labels were in fact period-end, only 30 minutes of
    lead time would be lost (conservative choice, no leakage).
    TODO: confirm the convention in the Meteocat documentation.
    """
    out = df.copy()
    out["timestamp"] = out["timestamp"] + pd.Timedelta(minutes=minutes)
    return out


def obs_to_wide(
    obs: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    freq: str = FREQ,
) -> pd.DataFrame:
    """Long format -> wide table on a regular grid, one column per series and aggregation."""
    cols = {}
    for (source, station, variable), part in obs.groupby(["source", "station", "variable"]):
        s = part.set_index("timestamp")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        for how in AGGREGATIONS.get(str(variable), ["mean"]):
            cols[column_name(source, station, variable, how)] = to_grid(s, how, freq)
    wide = pd.DataFrame(cols)
    if wide.empty:
        return wide
    lo = start if start is not None else wide.index.min()
    hi = end if end is not None else wide.index.max()
    grid = pd.date_range(lo.ceil(freq), hi.floor(freq), freq=freq, tz="UTC")
    return wide.reindex(grid)


def rain_intensity_to_mm(mean_intensity_mm_h: pd.Series, freq: str = FREQ) -> pd.Series:
    """Convert the mean intensity of an interval (mm/h) to precipitation (mm)."""
    return mean_intensity_mm_h * (pd.Timedelta(freq) / pd.Timedelta(hours=1))


def splice(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """Merge two series from the same site: the new one (primary) takes precedence and
    the old one (secondary) fills the time steps where the new one has no data."""
    return primary.combine_first(secondary)

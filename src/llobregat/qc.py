"""Quality control: flag suspicious values, without deleting them.

The same rules apply to the history and to real time. The target is never imputed; only
short gaps in predictor variables are interpolated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from llobregat.config import QCConfig


def run_lengths(values: pd.Series) -> pd.Series:
    """Length of the run of identical consecutive values that each point belongs to."""
    changed = values.ne(values.shift()) | values.isna()
    group = changed.cumsum()
    return values.groupby(group).transform("size")


def qc_series(s: pd.Series, cfg: QCConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Apply the quality rules to a regular time series.

    Returns the clean series (suspicious values set to NaN) and a table of boolean flags.
    """
    s = s.astype(float)
    flags = pd.DataFrame(index=s.index)
    out = s < cfg.min_value
    if cfg.max_value is not None:
        out |= s > cfg.max_value
    flags["out_of_range"] = out.fillna(False)

    runs = run_lengths(s)
    flags["flatline"] = ((runs >= cfg.flatline_intervals) & (s > cfg.flatline_min_value)).fillna(
        False
    )

    if cfg.max_jump is not None:
        jump = s.diff().abs() > cfg.max_jump
        flags["jump"] = jump.fillna(False)
    else:
        flags["jump"] = False

    bad = flags.any(axis=1)
    return s.mask(bad), flags


def fill_short_gaps(s: pd.Series, max_intervals: int) -> pd.Series:
    """Linearly interpolate only the gaps of at most `max_intervals` values.

    Longer gaps and those at the edges stay NaN.
    """
    isna = s.isna()
    if not isna.any():
        return s
    group = (~isna).cumsum()
    gap_len = isna.groupby(group).transform("sum").where(isna, 0)
    interpolated = s.interpolate(method="time", limit_area="inside")
    fill = isna & (gap_len <= max_intervals)
    return s.where(~fill, interpolated)


def qc_summary(flags: pd.DataFrame) -> pd.Series:
    """Percentage of values flagged by each rule."""
    if flags.empty:
        return pd.Series(dtype=float)
    return (flags.mean() * 100).round(3)


def remove_duplicates(
    df: pd.DataFrame, keys: list[str], time_col: str = "timestamp"
) -> pd.DataFrame:
    """Drop duplicates (keeping the last received) and sort by time."""
    return df.drop_duplicates(subset=keys, keep="last").sort_values(time_col).reset_index(drop=True)


def is_regular(index: pd.DatetimeIndex, freq: str = "30min") -> bool:
    if len(index) < 2:
        return True
    diffs = np.diff(index.as_unit("ns").asi8)
    return bool((diffs == pd.Timedelta(freq).value).all())

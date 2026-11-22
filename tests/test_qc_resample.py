import numpy as np
import pandas as pd

from llobregat.config import QCConfig
from llobregat.qc import fill_short_gaps, qc_series
from llobregat.resample import obs_to_wide, to_grid


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="30min", tz="UTC")


def test_qc_flags():
    s = pd.Series([1, 2, -1, 3, 9999, 4] + [7.0] * 15 + [8, 9], index=_idx(23), dtype=float)
    clean, flags = qc_series(s, QCConfig(max_value=1000, flatline_intervals=12))
    assert flags["out_of_range"].sum() == 2
    assert flags["flatline"].sum() == 15
    assert clean.isna().sum() == 17


def test_fill_short_gaps_only():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan], index=_idx(8))
    out = fill_short_gaps(s, max_intervals=2)
    assert out.iloc[1] == 2.0
    assert out.iloc[3:6].isna().all()  # gap of 3 > 2: not filled
    assert np.isnan(out.iloc[7])  # extrem: no s'omple


def test_grid_is_right_labelled():
    idx = pd.date_range("2020-01-01 00:05", periods=6, freq="5min", tz="UTC")
    s = pd.Series([1.0, 2, 3, 4, 5, 6], index=idx)
    g = to_grid(s, "mean")
    # Values from 00:05 to 00:30 → interval (00:00, 00:30], labelled 00:30
    assert g.index[0] == pd.Timestamp("2020-01-01 00:30", tz="UTC")
    assert g.iloc[0] == 3.5


def test_obs_to_wide():
    t = pd.date_range("2020-01-01 00:05", periods=12, freq="5min", tz="UTC")
    obs = pd.DataFrame(
        {
            "source": "aca",
            "station": "P/s",
            "variable": "flow",
            "timestamp": t,
            "value": np.arange(12.0),
            "flag": None,
        }
    )
    wide = obs_to_wide(obs)
    assert list(wide.columns) == ["aca:P/s:flow:mean", "aca:P/s:flow:max"]
    assert wide["aca:P/s:flow:max"].iloc[0] == 5


def test_splice_prefers_new_station():
    from llobregat.resample import splice

    new = pd.Series([np.nan, np.nan, 3.0, 4.0], index=_idx(4))
    old = pd.Series([1.0, 2.0, 9.0, np.nan], index=_idx(4))
    assert list(splice(new, old)) == [1.0, 2.0, 3.0, 4.0]


def test_rain_intensity_to_mm():
    from llobregat.resample import rain_intensity_to_mm

    assert rain_intensity_to_mm(pd.Series([6.0])).iloc[0] == 3.0

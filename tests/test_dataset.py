import numpy as np
import pandas as pd

from llobregat import dataset as ds


def test_previous_day_is_closed_day():
    daily = pd.Series(
        [1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    )
    grid = pd.DatetimeIndex(
        ["2020-01-02 00:00", "2020-01-02 00:30", "2020-01-03 00:00", "2020-01-03 12:00"], tz="UTC"
    )
    out = ds.previous_day(daily, grid)
    # 2020-01-02 00:00 closes an interval of 1 January: the previous complete day is 31-12
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1.0  # interval of day 2 -> value of day 1
    assert out.iloc[2] == 1.0
    assert out.iloc[3] == 2.0


def test_rain_aca_30min_sums_right_labelled(tmp_path):
    (tmp_path / "aca_rain").mkdir()
    t = pd.date_range("2020-01-01 00:05", periods=12, freq="5min", tz="UTC")
    v = [0.1] * 6 + [0.2] * 5 + [99.0]  # the last value is impossible
    pd.DataFrame({"timestamp": t, "value": v}).to_parquet(tmp_path / "aca_rain" / "EA049.parquet")
    s = ds.rain_aca_30min(tmp_path, "EA049")
    assert abs(s.loc["2020-01-01 00:30"] - 0.6) < 1e-9
    assert abs(s.loc["2020-01-01 01:00"] - 1.0) < 1e-9


def test_flow_30min_uses_clean_values(tmp_path):
    (tmp_path / "aca_flow_qc").mkdir()
    t = pd.date_range("2020-01-01 00:05", periods=6, freq="5min", tz="UTC")
    pd.DataFrame(
        {"timestamp": t, "value": [1, 2, 3, 4, 5, 600.0], "value_qc": [1, 2, 3, 4, 5, np.nan]}
    ).to_parquet(tmp_path / "aca_flow_qc" / "1.parquet")
    s = ds.flow_30min(tmp_path, "1")
    assert s.loc["2020-01-01 00:30"] == 3.0

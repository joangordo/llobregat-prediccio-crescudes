import numpy as np
import pandas as pd

from llobregat import qc_report as qr
from llobregat import variants as vr
from llobregat.config import ModelVariant


def test_coverage_30min_and_daily():
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-03", tz="UTC")
    t = pd.date_range("2020-01-01", "2020-01-02", freq="5min", tz="UTC", inclusive="left")
    assert abs(vr.coverage(t, start, end) - 0.5) < 1e-9
    d = pd.date_range("2020-01-01", periods=1, freq="D", tz="UTC")
    assert vr.coverage(d, start, end, daily=True) == 0.5
    assert vr.coverage(pd.DatetimeIndex([], tz="UTC"), start, end) == 0.0


def test_select_inputs_by_coverage():
    full = pd.date_range("2018-10-01", "2020-10-01", freq="30min", tz="UTC", inclusive="left")
    half = full[: len(full) // 2]
    cands = [vr.Candidate("flow", "a", full), vr.Candidate("flow", "b", half)]
    v = ModelVariant(name="x", train_start="2018-10-01", min_coverage=0.85)
    sel = vr.select_inputs(cands, v, end=pd.Timestamp("2020-10-01", tz="UTC"))
    assert sel.set_index("name")["selected"].to_dict() == {"a": 1, "b": 0}


def test_complete_years_only_drops_current_year():
    idx = pd.date_range("2024-10-01", "2026-11-20", freq="30min", tz="UTC")
    out = vr.complete_years_only(idx, 10)
    assert out.max() < pd.Timestamp("2026-10-01", tz="UTC")


def test_isolated_spikes_and_flat_runs():
    v = np.array([10.0, 11.0, 80.0, 12.0, 12.0, 13.0])
    assert qr.isolated_spikes(v).tolist() == [False, False, True, False, False, False]
    s = pd.Series(
        [5.0, 7.0, 7.0, 7.0, 0.0, 0.0, 0.0],
        index=pd.date_range("2020-01-01", periods=7, freq="1h", tz="UTC"),
    )
    runs = qr.flat_runs(s, min_value=0.5)
    assert len(runs) == 1 and runs["hours"].iloc[0] == 2.0 and runs["value"].iloc[0] == 7.0


def test_top_events_are_separated():
    idx = pd.date_range("2020-01-01", periods=24 * 20, freq="1h", tz="UTC")
    g = pd.Series(1.0, index=idx)
    g.iloc[10], g.iloc[11], g.iloc[24 * 10] = 100.0, 99.0, 50.0
    ev = qr.top_events(g, n=2, sep_days=3)
    assert [v for _, v in ev] == [100.0, 50.0]

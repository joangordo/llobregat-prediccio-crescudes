import numpy as np
import pandas as pd

from llobregat.config import FlowQCConfig
from llobregat.qc_flow import apply_flow_qc

P = FlowQCConfig()


def _df(values, start="2020-01-01", freq="5min", factor=None):
    t = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    d = pd.DataFrame({"timestamp": t, "value": np.asarray(values, dtype=float)})
    if factor is not None:
        d["unit_factor"] = factor
    return d


def _base(n=288 * 20, level=6.0, seed=0):
    rng = np.random.default_rng(seed)
    # Base flow always positive (the real series have no negatives)
    return level * (1 + 0.05 * np.sin(np.arange(n) / 300) + rng.normal(0, 0.002, n))


def flagged(out, rule):
    return out["flags"].str.contains(rule, regex=False).to_numpy()


def test_isolated_spikes_like_balsareny_2008():
    v = _base()
    v[3000], v[3002] = 909.98, 1140.82  # two spikes separated by a normal reading
    out, _ = apply_flow_qc(_df(v), None, P)
    assert flagged(out, "spike")[[3000, 3002]].all()
    assert flagged(out, "spike").sum() == 2


def test_flash_flood_rising_limb_is_kept():
    # Gavarresa 2022-07-29 (already in m³/s): 0.001 to 5 m³/s in 5 min, 1.5 h high, then down
    v = np.r_[
        np.full(2000, 0.001),
        np.linspace(2.3, 5.0, 18),
        np.geomspace(2.5, 0.2, 30),
        np.full(2000, 0.2),
    ]
    out, _ = apply_flow_qc(_df(v), None, P)
    assert out["flags"].eq("").all()


def test_smooth_big_flood_is_kept():
    # Castellbell 1996-11-17: smooth rise and fall up to ~970 m³/s
    x = np.linspace(-3, 3, 400)
    v = np.r_[_base(3000, 7.0), 7 + 960 * np.exp(-(x**2)), _base(3000, 9.0, seed=1)]
    out, _ = apply_flow_qc(_df(v), None, P)
    assert out["flags"].eq("").all()


def test_oscillation_like_mediona_2026():
    v = _base(level=0.02)
    v[5000:5024:2] = 230.0  # 230 and 0.02 alternating for 2 h
    out, _ = apply_flow_qc(_df(v), None, P)
    assert flagged(out, "oscillation")[5000:5024].all()


def test_saturated_and_stuck_sensors():
    v = _base(n=288 * 30, level=1.0)
    v[4000 : 4000 + 12 * 7] = 10.67  # 7 flat hours at high flow (Monegal 2019-12-20)
    v[6000 : 6000 + 288 * 8] = 2.5  # 8 days with the same value
    out, _ = apply_flow_qc(_df(v), None, P)
    assert flagged(out, "high_flat")[4000:4084].all()
    assert flagged(out, "stuck")[6000 : 6000 + 288 * 8].all()


def test_daily_balance_against_official_daily():
    v = _base(n=288 * 5, level=5.3)
    v[288 * 2 : 288 * 3] = 104.0 + np.linspace(0, 1, 288)  # one shifted day (Castellar 2019)
    d = _df(v)
    days = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    daily = pd.DataFrame({"timestamp": days, "value": 5.3})
    out, bal = apply_flow_qc(d, daily, P)
    day = out["timestamp"].dt.floor("D")
    assert flagged(out, "daily_balance")[(day == days[2]).to_numpy()].all()
    assert not flagged(out, "daily_balance")[(day != days[2]).to_numpy()].any()
    assert len(bal) == 1


def test_unit_step_flags_both_days():
    v = _base(n=288 * 4, level=5.0)
    f = np.r_[np.full(288 * 2, 1.0), np.full(288 * 2, 0.001)]
    v[288 * 2 :] = 0.3  # at midnight the factor changes and the value drops x16 (not enough)
    out, _ = apply_flow_qc(_df(v, factor=f), None, P)
    assert not flagged(out, "unit_step").any()
    v[288 * 2 :] = 0.005  # x1000: both days of the change are flagged
    out, _ = apply_flow_qc(_df(v, factor=f), None, P)
    assert flagged(out, "unit_step")[288 : 288 * 3].all()
    assert not flagged(out, "unit_step")[:288].any()


def test_exclusions_and_value_qc():
    v = _base(n=288 * 2)
    a, b = pd.Timestamp("2020-01-01 10:00", tz="UTC"), pd.Timestamp("2020-01-01 12:00", tz="UTC")
    out, _ = apply_flow_qc(_df(v), None, P, exclusions=[(a, b)])
    m = flagged(out, "exclusion")
    assert m.sum() == 25 and out.loc[m, "value_qc"].isna().all()
    assert out.loc[~m, "value_qc"].notna().all()


def test_excursions_like_merles_and_mediona():
    v = _base(n=288 * 30, level=0.16)
    v[300:312] = 127.0  # Merlès 2017-09-27: 1 h at 127 and back to 0.16
    v[600:612] = 230.0  # Mediona 2026-10-03
    out, _ = apply_flow_qc(_df(v), None, P)
    ex = flagged(out, "excursion")
    assert ex[300:312].all() and ex[600:612].all()
    assert ex.sum() == 24


def test_burst_like_balsareny_1997():
    v = _base(n=288 * 30, level=2.0)
    v[400:407] = [584.7, 1.78, 255.5, 1.96, 621.8, 1.96, 1.78]
    out, _ = apply_flow_qc(_df(v), None, P)
    assert out.loc[400:405, "value_qc"].isna().sum() >= 3
    assert out.loc[[400, 402, 404], "value_qc"].isna().all()


def test_real_flash_flood_like_vilada_2026_is_kept():
    peak = [
        0.66,
        1.15,
        40.0,
        152.4,
        194.2,
        160.0,
        105.2,
        98.0,
        91.9,
        80.0,
        63.7,
        58.0,
        50.5,
        45.0,
        39.5,
        36.0,
        33.3,
        30.0,
        27.0,
        24.0,
        22.0,
        20.0,
        18.0,
        16.0,
    ]
    v = np.r_[
        _base(2000, 0.12),
        peak,
        np.geomspace(15, 2, 200),
        np.full(500, 2.0) + np.linspace(0, 0.01, 500),
    ]
    out, _ = apply_flow_qc(_df(v), None, P)
    assert out["flags"].eq("").all()


def test_flood_that_recedes_in_days_is_not_an_excursion():
    # Regression: with pandas 3 the index is in µs and the 2 h window became 2000 h
    x = np.linspace(0, 1, 36)
    rise = 0.2 + 120 * x
    fall = 120 * np.exp(-np.linspace(0, 7, 288 * 2)) + 0.2
    v = np.r_[_base(2000, 0.2), rise, fall, _base(2000, 0.2, seed=2)]
    out, _ = apply_flow_qc(_df(v), None, P)
    assert not flagged(out, "excursion").any()


def test_oscillation_does_not_spread_to_hourly_data():
    t1 = pd.date_range("2009-12-01", periods=100, freq="1h", tz="UTC")
    t2 = pd.date_range(t1[-1] + pd.Timedelta("5min"), periods=48, freq="5min", tz="UTC")
    t3 = pd.date_range(t2[-1] + pd.Timedelta("1h"), periods=100, freq="1h", tz="UTC")
    t = t1.append(t2).append(t3)
    v = np.r_[np.linspace(0.5, 0.6, 100), np.tile([0.05, 180.0], 24), np.linspace(0.5, 0.6, 100)]
    out, _ = apply_flow_qc(pd.DataFrame({"timestamp": t, "value": v}), None, P)
    osc = flagged(out, "oscillation")
    assert osc[100:148].all()
    assert not osc[:99].any() and not osc[150:].any()


def test_zero_block_in_flood_like_guardiola_1997():
    flood = np.r_[np.linspace(140, 255, 36), np.zeros(36), np.linspace(252, 150, 48)]
    v = np.r_[_base(3000, 3.0), flood, np.geomspace(150, 3, 300), _base(3000, 3.0, seed=3)]
    out, _ = apply_flow_qc(_df(v), None, P)
    z = slice(3036, 3072)
    assert flagged(out, "dropout")[z].all()
    assert not flagged(out, "excursion").any()
    assert out["value_qc"].iloc[3072:3120].notna().all()


def test_repeated_constant_blocks_like_mediona_2026():
    # Exact 230.00 blocks of 1 h separated by 0.02: the blocks are the error, not the zeros
    v = _base(n=288 * 30, level=0.02)
    for s in (5000, 5024, 5048):
        v[s : s + 12] = 230.0
    out, _ = apply_flow_qc(_df(v), None, P)
    hi = np.r_[5000:5012, 5024:5036, 5048:5060]
    lo = np.r_[5012:5024, 5036:5048]
    assert out["value_qc"].iloc[hi].isna().all()
    assert not flagged(out, "dropout")[lo].any()


def test_excursion_after_a_gap_like_cardona_2007():
    t = pd.date_range("2007-01-01", periods=288 * 30, freq="5min", tz="UTC")
    v = _base(n=len(t), level=3.2)
    d = pd.DataFrame({"timestamp": t, "value": v})
    # 1.5 h gap, then 214, 408, 83 and back to 3.26 within 2 h
    gap = (d["timestamp"] > "2007-01-20 07:30") & (d["timestamp"] < "2007-01-20 09:00")
    d = d[~gap].copy()
    for ts, val in [("09:00", 214.4), ("09:55", 408.9), ("10:00", 83.4)]:
        d.loc[d["timestamp"] == pd.Timestamp(f"2007-01-20 {ts}", tz="UTC"), "value"] = val
    out, _ = apply_flow_qc(d, None, P)
    m = out["timestamp"] == pd.Timestamp("2007-01-20 09:00", tz="UTC")
    assert out.loc[m, "value_qc"].isna().all()


def test_constant_blocks_with_interleaved_zeros_like_mediona_2026():
    v = _base(n=288 * 30, level=0.02)
    v[5000:5024] = 230.0
    v[[5006, 5015]] = 0.02  # zeros interleaved within the block
    out, _ = apply_flow_qc(_df(v), None, P)
    assert out["value_qc"].iloc[5000:5024].isna().all()
    assert not flagged(out, "dropout")[5024:5100].any()

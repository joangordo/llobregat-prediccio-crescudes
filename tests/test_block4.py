"""Calibrated intervals (CQR on a log scale, ACI) and warnings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llobregat import analysis
from llobregat import probabilistic as P
from test_block3 import setup  # noqa: F401  (shared fixture)


def _hetero(n, scale, seed, start="2020-01-01"):
    """Flow with multiplicative error: the quantile "model" is too narrow (×0.5)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    med = np.exp(rng.normal(2, 1, n))
    obs = med * np.exp(rng.normal(0, scale, n))
    z = 1.2816 * 0.5 * 0.4  # 10 and 90 % quantiles with half the real spread (0.4)
    return pd.DataFrame(
        {"obs": obs, "q_now": med, "q10": med * np.exp(-z), "q90": med * np.exp(z)}, index=t
    )


def test_cqr_log_restores_coverage():
    cal, test = _hetero(4000, 0.4, 0), _hetero(4000, 0.4, 1)
    raw = np.mean((test["obs"] >= test["q10"]) & (test["obs"] <= test["q90"]))
    lo, hi, (a_lo, a_hi) = P.cqr_log(cal, test, "q10", "q90", 0.2)
    cov = np.mean((test["obs"] >= lo) & (test["obs"] <= hi))
    assert raw < 0.6 and abs(cov - 0.8) < 0.03
    assert a_lo > 0 and a_hi > 0  # the interval widens
    # the adjustment is multiplicative (except for the +0.1 of the log scale): the relative
    # width hardly depends on the flow level
    rel = (hi - lo) / test["q_now"]
    assert rel.std() / rel.mean() < 0.05


def test_aci_adapts_to_a_shift_and_respects_the_delay():
    cal = _hetero(3000, 0.4, 0)
    shifted = _hetero(6000, 0.7, 2)  # larger errors than in calibration
    lo_c, hi_c, _ = P.cqr_log(cal, shifted, "q10", "q90", 0.2)
    cov_cqr = np.mean((shifted["obs"] >= lo_c) & (shifted["obs"] <= hi_c))
    lo, hi, alphas = P.aci_log(cal, shifted, "q10", "q90", 0.2, gamma=0.01, h=6)
    cov_aci = np.mean((shifted["obs"] >= lo) & (shifted["obs"] <= hi))
    assert cov_cqr < 0.7 and abs(cov_aci - 0.8) < abs(cov_cqr - 0.8)
    assert abs(cov_aci - 0.8) < 0.05
    # During the first 6 h nothing has been observed yet: α does not move
    assert np.all(alphas[:12] == 0.2) and alphas[13] != 0.2


def test_persistence_conformal():
    cal, test = _hetero(4000, 0.4, 0), _hetero(4000, 0.4, 1)
    lo, hi = P.persistence_conformal(cal, test, 0.2)
    cov = np.mean((test["obs"] >= lo) & (test["obs"] <= hi))
    assert abs(cov - 0.8) < 0.03


def test_weighted_interval_score():
    obs = np.array([1.0, 2.0, 3.0])
    w = P.weighted_interval_score(obs, obs, {0.2: (obs - 1, obs + 1)})
    assert w == pytest.approx((0.2 / 2 * 2) / 1.5)  # perfect median, interval of width 2
    worse = P.weighted_interval_score(obs + 5, obs, {0.2: (obs - 1, obs + 1)})
    assert worse > w


def test_choose_rule_uses_calibration_events():
    t = pd.date_range("2020-01-01", periods=400, freq="30min", tz="UTC")
    obs = pd.Series(10.0, index=t)
    obs.iloc[200:220] = 200
    cal = pd.DataFrame({"obs": obs, "central": 10.0, "upper90": 10.0}, index=t)
    cal["upper80"] = obs.shift(-0).to_numpy()  # the 80 % bound sees the flood
    assert P.choose_rule([(cal, 100.0)], h=1, step_h=0.5) == "upper80"
    quiet = cal.assign(obs=10.0)
    assert P.choose_rule([(quiet, 100.0)], h=1, step_h=0.5) == P.DEFAULT_RULE


def test_intervals_end_to_end_and_final_lock(setup):  # noqa: F811
    cfg, variant, root, data = setup
    analysis.run(cfg, variant, root, sample=40_000)
    from llobregat.experiment import dataset_dir, dataset_folds

    folds = [f for f in dataset_folds(data, cfg, variant) if f.name != "final"]
    out = dataset_dir(root, cfg, variant) / "uncertainty"
    res = P.run(cfg, variant, root, folds, [1], out, max_rounds=60, rapid=True)
    pred, met, al = res["predictions"], res["metrics"], res["warnings"]
    for c in ("central", "q05", "q95", "lo80_cqr", "hi80_cqr", "lo90_aci", "hi90_aci"):
        assert c in pred
    assert (pred["q05"] <= pred["q10"]).all() and (pred["q90"] <= pred["q95"]).all()
    assert (pred["lo80_cqr"] <= pred["hi80_cqr"]).all()
    cqr = met[(met["method"] == "cqr") & (met["level"] == 80)]["coverage"]
    assert cqr.between(0.6, 0.97).all()
    assert set(al["rule"]) == set(P.RULES) and al.groupby("fold")["chosen"].sum().eq(1).all()
    assert set(pred["hydro_year"]) == {2015, 2016}  # never the hold-out
    assert (out / "summary.txt").exists()

    # Final evaluation: once only
    fin = P.final_evaluation(cfg, variant, root, [1], max_rounds=60, rapid=True)
    assert set(fin["predictions"]["hydro_year"]) == {2017}
    with pytest.raises(SystemExit, match="already run"):
        P.final_evaluation(cfg, variant, root, [1], max_rounds=60, rapid=True)
    lock = dataset_dir(root, cfg, variant) / "final" / "HOLDOUT_EVALUATED.json"
    assert lock.exists()

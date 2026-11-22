import numpy as np

from llobregat.intervals import CQR, AdaptiveConformal, MondrianCQR, conformal_quantile, rearrange
from llobregat.metrics import coverage


def _data(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = x + rng.normal(0, 1 + 0.3 * x, n) * scale
    # Deliberately too narrow quantiles: CQR must correct them
    return y, x - 0.5, x + 0.5


def test_conformal_quantile_finite_sample():
    assert conformal_quantile(np.arange(1, 10), 0.8) == 8
    assert conformal_quantile(np.arange(1, 5), 0.99) == np.inf


def test_cqr_marginal_coverage_iid():
    y_cal, lo_cal, hi_cal = _data(3000, 0)
    y_te, lo_te, hi_te = _data(20000, 1)
    for asym in (False, True):
        lo, hi = CQR(alpha=0.2, asymmetric=asym).fit(y_cal, lo_cal, hi_cal).predict(lo_te, hi_te)
        assert 0.78 <= coverage(y_te, lo, hi) <= 0.83


def test_mondrian_uses_groups():
    y_cal, lo_cal, hi_cal = _data(3000, 0)
    groups = np.where(lo_cal > 6, "flood", "normal")
    model = MondrianCQR(alpha=0.2, min_n=100).fit(y_cal, lo_cal, hi_cal, groups)
    assert set(model.by_group) == {"flood", "normal"}
    y_te, lo_te, hi_te = _data(20000, 1)
    g_te = np.where(lo_te > 6, "flood", "normal")
    lo, hi = model.predict(lo_te, hi_te, g_te)
    for g in ("flood", "normal"):
        assert 0.76 <= coverage(y_te[g_te == g], lo[g_te == g], hi[g_te == g]) <= 0.85


def test_aci_long_run_coverage_under_shift():
    y_cal, lo_cal, hi_cal = _data(2000, 0)
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    aci = AdaptiveConformal(scores, alpha=0.2, gamma=0.01)
    y_te, lo_te, hi_te = _data(20000, 2, scale=2.0)  # distribution shift: errors doubled
    for y, lo, hi in zip(y_te, lo_te, hi_te, strict=True):
        aci.update(y, lo, hi)
    assert abs(aci.long_run_error() - 0.2) < 0.02


def test_rearrange():
    lo, _mid, hi = rearrange([3, 1], [2, 2], [1, 3])
    assert list(lo) == [1, 1] and list(hi) == [3, 3]

import numpy as np
import pytest

from llobregat import metrics as m


def test_perfect_prediction():
    y = np.array([1.0, 3.0, 2.0, 8.0, 5.0])
    assert m.nse(y, y) == pytest.approx(1)
    assert m.kge(y, y)["kge"] == pytest.approx(1)
    assert m.rmse(y, y) == 0


def test_mean_prediction_has_nse_zero():
    y = np.array([1.0, 3.0, 2.0, 8.0, 5.0])
    assert m.nse(y, np.full_like(y, y.mean())) == pytest.approx(0)


def test_persistence_index():
    y = np.array([1.0, 2, 3, 4, 5])
    pers = np.array([0.0, 1, 2, 3, 4])
    assert m.persistence_index(y, pers, pers) == pytest.approx(0)
    assert m.persistence_index(y, y, pers) == pytest.approx(1)


def test_event_scores():
    obs = np.array([0, 0, 5, 9, 4, 0, 0, 0, 7, 0, 0, 0], dtype=float)
    alert = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    s = m.event_scores(obs, alert, threshold=3, lead_steps=1)
    assert (s["hits"], s["misses"], s["false_alarms"]) == (1, 1, 1)
    assert s["csi"] == pytest.approx(1 / 3)


def test_interval_metrics():
    y = np.array([1.0, 2, 3, 10])
    lo, hi = np.zeros(4), np.full(4, 5.0)
    assert m.coverage(y, lo, hi) == pytest.approx(0.75)
    assert m.interval_score(y, lo, hi, alpha=0.2) == pytest.approx(5 + (2 / 0.2) * 5 / 4)
    assert m.pinball(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)


def test_block_bootstrap_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, 2000)
    good, bad = y + rng.normal(0, 0.5, 2000), y + rng.normal(0, 2, 2000)
    groups = np.repeat(np.arange(10), 200)
    res = m.block_bootstrap_diff(m.rmse, y, good, bad, groups, n_boot=200)
    assert res["ci_high"] < 0


def test_diebold_mariano():
    rng = np.random.default_rng(0)
    la, lb = rng.normal(1, 0.1, 500), rng.normal(1.5, 0.1, 500)
    assert m.diebold_mariano(la, lb, h=3)["p_value"] < 0.01

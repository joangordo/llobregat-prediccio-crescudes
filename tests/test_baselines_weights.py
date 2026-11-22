import numpy as np

from llobregat import weights
from llobregat.models.baselines import Climatology, Persistence


def test_persistence(basin):
    q = basin["aca:P/target:flow:mean"]
    model = Persistence(h_steps=6).fit(q)
    qs = model.predict_quantiles(q.to_numpy()[:10])
    assert np.all(qs[0.1] <= qs[0.5]) and np.all(qs[0.5] <= qs[0.9])
    np.testing.assert_allclose(model.predict(q[:3]), q[:3])


def test_climatology(basin):
    q = basin["aca:P/target:flow:mean"]
    model = Climatology().fit(q)
    pred = model.predict(q.index[:5], h_steps=2)
    assert len(pred) == 5 and np.all(np.isfinite(pred))


def test_weights_mean_one():
    q = np.array([1.0, 2, 3, 50, 80, 2, 1])
    for w in (
        weights.power(q, 1.0),
        weights.threshold(q, 40, 10),
        weights.event_window(q, 40, 10, before_steps=2),
        weights.uniform(q),
    ):
        assert np.isclose(w.mean(), 1)
    w = weights.event_window(q, 40, 10, before_steps=2)
    assert w[1] > w[0]  # the preceding rise also gets more weight


def test_tfg_stepped_matches_tfg_rule():
    q = np.arange(1.0, 101.0)  # Q75≈75.3 · Q90≈90.1 · Q95≈95.05 · Q99≈99.01
    w = weights.tfg_stepped(q)
    raw = w / w.min()  # undo the normalisation (the base weight is 1)
    assert raw[0] == 1 and raw[80] == 3 and raw[92] == 5 and raw[96] == 100

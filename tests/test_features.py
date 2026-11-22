import numpy as np
import pandas as pd

from llobregat.features import (
    antecedent_index,
    build_features,
    estimate_travel_time,
    inverse_target,
    lag_window,
    make_target,
)


def test_no_future_leakage(basin, spec):
    """Changing the data after t cannot change the features at t."""
    t = basin.index[5000]
    base = build_features(basin, spec).loc[:t]

    rng = np.random.default_rng(1)
    altered = basin.copy()
    future = altered.index > t
    altered.loc[future] = rng.uniform(0, 1000, size=(future.sum(), altered.shape[1]))
    after = build_features(altered, spec).loc[:t]

    pd.testing.assert_frame_equal(base, after)


def test_feature_names_and_shape(basin, spec):
    f = build_features(basin, spec)
    assert len(f) == len(basin)
    for col in [
        "q_now",
        "q_lag_1h",
        "up_up_lag_3h",
        "rain_sb_nord_sum_6h",
        "rain_basin_sum_72h",
        "api_kd0p9",
        "hours_since_rain",
        "doy_sin",
    ]:
        assert col in f.columns


def test_target_roundtrip():
    q = pd.Series([1.0, 2.0, 5.0, 3.0, 0.0, 4.0])
    for transform in ["level", "delta", "log_ratio", "log_level"]:
        y = make_target(q, 2, transform)
        back = inverse_target(y[:-2], q[:-2], transform)
        np.testing.assert_allclose(back, q.shift(-2)[:-2], atol=1e-9)


def test_antecedent_index_decays():
    p = pd.Series([10.0] + [0.0] * 5)
    api = antecedent_index(p, 0.5)
    np.testing.assert_allclose(api.to_numpy(), [10, 5, 2.5, 1.25, 0.625, 0.3125])


def test_travel_time_recovered(basin):
    lag, corr = estimate_travel_time(
        basin["aca:P/up:flow:mean"], basin["aca:P/target:flow:mean"], 20
    )
    assert lag == 8
    assert corr.idxmax() == 8


def test_lag_window():
    assert lag_window(6, 10, 1, 2) == (3, 11)
    assert lag_window(6, 10, 12, 2) == (0, 0)

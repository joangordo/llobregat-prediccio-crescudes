"""Synthetic data to test the pipeline without real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llobregat.features import FeatureSpec


def synthetic_basin(start="2015-10-01", years=1.0, seed=0, travel_steps=8) -> pd.DataFrame:
    """Synthetic basin: rain at 3 gauges, upstream flow and flow at the target station."""
    rng = np.random.default_rng(seed)
    n = int(years * 365 * 48)
    idx = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    rain = np.zeros((n, 3))
    for _ in range(int(40 * years)):  # rain episodes
        t0 = rng.integers(0, n - 48)
        dur = rng.integers(4, 24)
        rain[t0 : t0 + dur] += rng.gamma(2, 1.5, size=(dur, 3))
    basin_rain = rain.mean(axis=1)
    kernel = np.exp(-np.arange(200) / 30)
    runoff = np.convolve(basin_rain, kernel)[:n]
    upstream = 5 + 3 * runoff + rng.normal(0, 0.3, n)
    target = np.roll(upstream, travel_steps) * 1.5 + rng.normal(0, 0.3, n)
    target[:travel_steps] = target[travel_steps]
    return pd.DataFrame(
        {
            "aca:P/target:flow:mean": np.clip(target, 0, None),
            "aca:P/up:flow:mean": np.clip(upstream, 0, None),
            "xema:AA:35:sum": rain[:, 0],
            "xema:BB:35:sum": rain[:, 1],
            "xema:CC:35:sum": rain[:, 2],
        },
        index=idx,
    )


@pytest.fixture
def basin() -> pd.DataFrame:
    return synthetic_basin()


@pytest.fixture
def spec() -> FeatureSpec:
    return FeatureSpec(
        target_col="aca:P/target:flow:mean",
        upstream={"up": ("aca:P/up:flow:mean", [0, 1, 2, 3, 4, 6])},
        rain_cols={"AA": "xema:AA:35:sum", "BB": "xema:BB:35:sum", "CC": "xema:CC:35:sum"},
        subbasins={"nord": ["AA", "BB"], "sud": ["CC"]},
    )

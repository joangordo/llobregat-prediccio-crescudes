import pytest
from pydantic import ValidationError

from llobregat.config import IntervalConfig, StationConfig, load_station_config


def test_station_yaml_is_valid():
    cfg = load_station_config("configs/stations/sant_joan_despi.yaml")
    assert cfg.horizons_h == [1, 3, 6, 12, 24]
    assert cfg.steps(6) == 12
    assert cfg.intervals.alpha == pytest.approx(0.2)


def test_horizon_must_fit_grid():
    with pytest.raises(ValidationError):
        StationConfig(
            station_id="x", name="x", target={"provider": "p", "sensor": "s"}, horizons_h=[0.25]
        )


def test_interval_config_consistency():
    with pytest.raises(ValidationError):
        IntervalConfig(alpha=0.2, quantiles=(0.05, 0.5, 0.95))

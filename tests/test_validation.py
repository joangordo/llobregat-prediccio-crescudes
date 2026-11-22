import numpy as np
import pandas as pd
import pytest

from llobregat.config import ValidationConfig
from llobregat.validation import folds_table, hydro_year, make_folds


@pytest.fixture
def index():
    return pd.date_range("2008-10-01", "2026-09-30 23:30", freq="30min", tz="UTC")


def test_hydro_year():
    idx = pd.DatetimeIndex(["2019-09-30", "2019-10-01", "2020-09-30"], tz="UTC")
    assert list(hydro_year(idx)) == [2019, 2020, 2020]


def test_folds_are_ordered_disjoint_and_gapped(index):
    cfg = ValidationConfig()
    folds = make_folds(index, cfg)
    assert len(folds) == cfg.n_outer_folds + 1
    assert folds[-1].name == "final" and folds[-1].test_years == [2025, 2026]
    gap = pd.Timedelta(days=cfg.gap_days)
    for f in folds:
        blocks = [f.train, f.calib, f.test]
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                assert not (blocks[i] & blocks[j]).any()
        # Temporal order and gap between blocks
        assert index[f.train].max() + gap <= index[f.calib].min()
        assert index[f.calib].max() + gap <= index[f.test].min()
        assert index[f.inner_train].max() + gap <= index[f.inner_val].min()
        # Never train with years after the test
        assert max(f.train_years) < min(f.test_years)
    print(folds_table(folds))


def test_holdout_never_used_before_final(index):
    folds = make_folds(index, ValidationConfig())
    holdout = np.isin(hydro_year(index), [2025, 2026])
    for f in folds[:-1]:
        assert not (holdout & (f.train | f.calib | f.test)).any()


def test_not_enough_years():
    idx = pd.date_range("2020-10-01", "2023-09-30", freq="30min", tz="UTC")
    with pytest.raises(ValueError):
        make_folds(idx, ValidationConfig())

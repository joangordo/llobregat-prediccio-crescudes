"""Common pieces of the experiments: reading the dataset and the validation folds.

The folds are the same as shown by `llobregat splits`: complete hydrological years with
an observed target, from the start of the variant. The masks are returned aligned with
the full index of the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import ModelVariant, StationConfig
from llobregat.validation import Fold, hydro_year, make_folds
from llobregat.variants import complete_years_only

TARGET_PREFIX = "y_"
OWN_FLOW = "q_now"


def dataset_dir(processed: Path, cfg: StationConfig, variant: ModelVariant) -> Path:
    return Path(processed) / cfg.station_id / variant.name


def load_dataset(
    processed: Path, cfg: StationConfig, variant: ModelVariant
) -> tuple[pd.DataFrame, dict]:
    d = dataset_dir(processed, cfg, variant)
    f = d / "dataset.parquet"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat build-dataset first")
    data = pd.read_parquet(f)
    meta_f = d / "meta.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    return data, meta


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if not c.startswith(TARGET_PREFIX)]


def target_column(h: float) -> str:
    return f"{TARGET_PREFIX}{h:g}h"


@dataclass
class FoldMasks:
    """A fold with its masks over the full index of the dataset."""

    fold: Fold
    train: np.ndarray
    inner_train: np.ndarray
    inner_val: np.ndarray
    calib: np.ndarray
    test: np.ndarray

    @property
    def name(self) -> str:
        return self.fold.name


def dataset_folds(data: pd.DataFrame, cfg: StationConfig, variant: ModelVariant) -> list[FoldMasks]:
    """Folds of the variant (with the hold-out as the last fold, 'final')."""
    vcfg = cfg.validation_for(variant)
    observed = data.index[data[OWN_FLOW].notna()]
    idx = complete_years_only(pd.DatetimeIndex(observed), vcfg.hydro_year_start_month)
    folds = make_folds(idx, vcfg)
    pos = data.index.get_indexer(idx)
    out = []
    for f in folds:
        masks = {}
        for name in ("train", "inner_train", "inner_val", "calib", "test"):
            m = np.zeros(len(data), dtype=bool)
            m[pos] = getattr(f, name)
            masks[name] = m
        out.append(FoldMasks(f, **masks))
    return out


def development_mask(folds: list[FoldMasks]) -> np.ndarray:
    """Development period: the training period of the first fold.

    It precedes every calibration and test year, so the exploratory analysis and the
    decisions taken from it see no evaluation year.
    """
    return folds[0].train.copy()


def hydro_years_of(index: pd.DatetimeIndex, cfg: StationConfig) -> np.ndarray:
    return hydro_year(index, cfg.validation.hydro_year_start_month)

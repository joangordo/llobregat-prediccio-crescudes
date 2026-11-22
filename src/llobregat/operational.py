"""Operational model: the frozen system retrained with all the data, saved to disk.

This is not the model that was evaluated: it is the same system (same variant, features,
LightGBM parameters, ensemble, interval method and warning rules) retrained with every
complete hydrological year. The last complete year is kept aside to calibrate the
intervals (CQR needs data the models have not seen); everything before it is training.

    uv run llobregat operational-model --variant long

Writes data/processed/<station>/<variant>/operational/:
    manifest.json             everything needed to reproduce a forecast (features, columns
                              per horizon, parameters, CQR adjustments, thresholds, periods)
    h<h>_lgbm.txt.gz          LightGBM point model (LightGBM text format, gzipped)
    h<h>_q05.txt.gz ...       LightGBM quantile models (q05, q10, q90, q95)
    h<h>_ridge.joblib         Ridge model (scikit-learn pipeline)
    source_importance.csv     grouped permutation importance (inner validation), per horizon
    feature_importance.csv    LightGBM gain importance of the point model, per horizon

`OperationalModel.load(directory).predict(features)` turns a feature table (one row per
issue time, built with llobregat.features exactly as in training) into forecasts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import ModelVariant, StationConfig, load_station_config
from llobregat.experiment import (
    OWN_FLOW,
    FoldMasks,
    dataset_dir,
    feature_columns,
    load_dataset,
    target_column,
)
from llobregat.validation import Fold, hydro_year
from llobregat.variants import complete_years_only

log = logging.getLogger("operational")

FORMAT_VERSION = 1
LEVELS = {80: 0.20, 90: 0.10}


# --------------------------------------------------------------------------- #
# Training split: everything for training except the last year (calibration)
# --------------------------------------------------------------------------- #
def operational_masks(data: pd.DataFrame, cfg: StationConfig, variant: ModelVariant) -> FoldMasks:
    vcfg = cfg.validation_for(variant)
    observed = data.index[data[OWN_FLOW].notna()]
    idx = complete_years_only(pd.DatetimeIndex(observed), vcfg.hydro_year_start_month)
    years_idx = hydro_year(idx, vcfg.hydro_year_start_month)
    uniq = sorted(np.unique(years_idx))
    calib_years = uniq[-vcfg.calibration_years :]
    train_years = uniq[: -vcfg.calibration_years]
    inner_val_years = train_years[-vcfg.inner_val_years :]
    gap = pd.Timedelta(days=vcfg.gap_days)
    start_of = {y: idx[years_idx == y].min() for y in uniq}

    def purge(mask, boundary):
        return mask & ~((idx >= boundary - gap) & (idx < boundary))

    calib = np.isin(years_idx, calib_years)
    train = purge(np.isin(years_idx, train_years), start_of[calib_years[0]])
    inner_val = np.isin(years_idx, inner_val_years) & train
    inner_train = purge(train & ~inner_val, start_of[inner_val_years[0]])
    fold = Fold(
        "operational",
        [],
        calib_years,
        train_years,
        inner_val_years,
        train,
        inner_train,
        inner_val,
        calib,
        np.zeros(len(idx), dtype=bool),
    )
    pos = data.index.get_indexer(idx)
    masks = {}
    for name in ("train", "inner_train", "inner_val", "calib", "test"):
        m = np.zeros(len(data), dtype=bool)
        m[pos] = getattr(fold, name)
        masks[name] = m
    return FoldMasks(fold, **masks)


# --------------------------------------------------------------------------- #
# Saving and loading
# --------------------------------------------------------------------------- #
def _save_lgbm(model, path: Path) -> dict:
    """LightGBM text model, gzipped (a few MB each instead of tens)."""
    text = model.booster.model_to_string(num_iteration=model.best_iteration)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return {
        "file": path.name,
        "features": list(model.features),
        "transform": model.transform,
        "eps": model.eps,
        "z_range": [float(model.z_range[0]), float(model.z_range[1])],
        "best_iteration": int(model.best_iteration),
    }


@dataclass
class _LoadedLGBM:
    booster: object
    features: list[str]
    transform: str
    eps: float
    z_range: tuple[float, float]

    def predict(self, X: pd.DataFrame, q_now) -> np.ndarray:
        from llobregat.features import inverse_target

        z = self.booster.predict(X[self.features].to_numpy(dtype=np.float32))
        z = np.clip(z, *self.z_range)
        return inverse_target(z, q_now, self.transform, self.eps)


def _load_lgbm(directory: Path, info: dict) -> _LoadedLGBM:
    import lightgbm as lgb

    with gzip.open(directory / info["file"], "rt", encoding="utf-8") as fh:
        booster = lgb.Booster(model_str=fh.read())
    return _LoadedLGBM(
        booster, info["features"], info["transform"], info["eps"], tuple(info["z_range"])
    )


def _log(x):
    from llobregat.probabilistic import to_log

    return to_log(x)


def _unlog(z):
    from llobregat.probabilistic import from_log

    return from_log(z)


@dataclass
class OperationalModel:
    directory: Path
    manifest: dict
    models: dict = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path | str) -> OperationalModel:
        import joblib

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported operational model format")
        models = {}
        for key, h_info in manifest["horizons"].items():
            models[key] = {
                "lgbm": _load_lgbm(directory, h_info["lgbm"]),
                "ridge": joblib.load(directory / h_info["ridge"]["file"]),
                "quantiles": {
                    tau: _load_lgbm(directory, info) for tau, info in h_info["quantiles"].items()
                },
            }
        return cls(directory, manifest, models)

    @property
    def horizons(self) -> list[float]:
        return [float(k) for k in self.manifest["horizons"]]

    @property
    def features(self) -> list[str]:
        return self.manifest["features"]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forecasts for every row of X (issue times) and every horizon.

        Columns: issue_time, horizon_h, target_time, q_now, central, lgbm, ridge, q05, q10,
        q90, q95, lo80, hi80, lo90, hi90, threshold_m3s, alert, pre_alert.
          alert      the point forecast is above the warning threshold
          pre_alert  the upper bound of the calibrated 90 % interval is above it
        """
        if OWN_FLOW not in X:
            raise KeyError(f"The feature table needs the column {OWN_FLOW}")
        q_now = X[OWN_FLOW].to_numpy(dtype=float)
        out = []
        for key, h_info in self.manifest["horizons"].items():
            m = self.models[key]
            h = float(key)
            df = pd.DataFrame(
                {"issue_time": X.index, "horizon_h": h, "q_now": q_now}, index=X.index
            )
            df["target_time"] = X.index + pd.Timedelta(hours=h)
            df["lgbm"] = m["lgbm"].predict(X, q_now)
            df["ridge"] = m["ridge"].predict(X[self.features], q_now)
            df["central"] = (df["lgbm"] + df["ridge"]) / 2
            taus = sorted(m["quantiles"], key=float)
            raw = np.sort(np.vstack([m["quantiles"][t].predict(X, q_now) for t in taus]), axis=0)
            for t, v in zip(taus, raw, strict=True):
                df[f"q{round(float(t) * 100):02d}"] = v
            for lev, alpha in LEVELS.items():
                lo_c = f"q{round(alpha / 2 * 100):02d}"
                hi_c = f"q{round((1 - alpha / 2) * 100):02d}"
                a_lo, a_hi = h_info["cqr"][str(lev)]
                df[f"lo{lev}"] = _unlog(_log(df[lo_c]) - a_lo)
                df[f"hi{lev}"] = _unlog(_log(df[hi_c]) + a_hi)
            thr = self.manifest["thresholds"]["warning_m3s"]
            df["threshold_m3s"] = thr
            df["alert"] = df["central"] > thr
            df["pre_alert"] = df["hi90"] > thr
            out.append(df)
        return pd.concat(out).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def input_ranges(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Quantiles of each raw input over the training period, from the features that are the
    input itself (current flow, current outflow and volume, last half hour of rain).

    The live app uses them to check units (e.g. a volume in m³ instead of hm³) and as the
    reference quantiles of the flow quality-control rules.
    """
    out = {}
    for col in train.columns:
        if col == OWN_FLOW:
            key = "q"
        elif col.startswith("up_") and col.endswith("_lag_0h"):
            key = "up:" + col[3:-7]
        elif col.startswith("res_") and col.endswith(("_out_now", "_vol_now")):
            name, kind = col[4:-8], col[-7:-4]
            key = f"res:{name}:{kind}"
        elif col.startswith("rain_st_") and col.endswith("_sum_0p5h"):
            key = "rain:" + col[8:-9]
        else:
            continue
        v = train[col].dropna()
        if len(v):
            qs = v.quantile([0.01, 0.5, 0.95, 0.99]).to_numpy()
            out[key] = {
                "p01": float(qs[0]),
                "p50": float(qs[1]),
                "p95": float(qs[2]),
                "p99": float(qs[3]),
                "max": float(v.max()),
            }
    return out


def _versions() -> dict[str, str]:
    """Library versions: the Ridge pipeline is pickled, so scikit-learn must match on load."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("llobregat-flood-forecasting", "lightgbm", "scikit-learn", "numpy", "pandas"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def train_operational(
    cfg: StationConfig,
    variant: ModelVariant,
    processed: Path,
    horizons: list[float] | None = None,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.90, 0.95),
    max_rounds: int = 3000,
    rapid: bool = False,
) -> Path:
    import joblib

    from llobregat.probabilistic import cqr_log, fit_models, lgbm_params_for, predict_frame
    from llobregat.train import selected_features, thresholds

    horizons = horizons or cfg.horizons_h
    data, meta = load_dataset(processed, cfg, variant)
    fm = operational_masks(data, cfg, variant)
    feats = selected_features(processed, cfg, variant, feature_columns(data), "selection")
    q = data[OWN_FLOW]
    thr = thresholds(q[fm.train].dropna(), cfg)
    out = dataset_dir(processed, cfg, variant) / "operational"
    out.mkdir(parents=True, exist_ok=True)
    f = fm.fold
    log.info(
        "%s: training %d–%d, inner validation %s, calibration %s, %d features",
        variant.name,
        f.train_years[0],
        f.train_years[-1],
        ",".join(map(str, f.inner_val_years)),
        ",".join(map(str, f.calib_years)),
        len(feats),
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "station": cfg.station_id,
        "variant": variant.name,
        "resolution_min": cfg.resolution_min,
        "train_years": [int(y) for y in f.train_years],
        "inner_val_years": [int(y) for y in f.inner_val_years],
        "calibration_years": [int(y) for y in f.calib_years],
        "data_end": str(data.index[data[OWN_FLOW].notna()].max()),
        "versions": _versions(),
        "inputs": meta.get("inputs") or meta.get("entrades") or {},
        "input_ranges": input_ranges(data[fm.train]),
        "features": feats,
        "thresholds": {
            "warning_m3s": thr["warning"],
            "warning_quantile": cfg.thresholds.warning_quantile,
            "high_flow_m3s": thr["high"],
        },
        "ensemble": {"lgbm": 0.5, "ridge": 0.5},
        "interval_method": "CQR (asymmetric, log scale) on the calibration year",
        "warning_rules": {
            "alert": "point forecast > warning threshold",
            "pre_alert": "upper bound of the 90 % interval > warning threshold",
        },
        "horizons": {},
    }
    src_imp, gain_imp = [], []
    for h in horizons:
        t0 = time.perf_counter()
        params = lgbm_params_for(processed, cfg, variant, h)
        if rapid:
            params["learning_rate"] = 0.1
        models = fit_models(data, fm, feats, h, params, quantiles, max_rounds)
        y = data[target_column(h)]
        base = (q.notna() & y.notna()).to_numpy()
        cal = predict_frame(models, data, fm.calib & base, h)
        key = f"{h:g}"
        info = {
            "lgbm": _save_lgbm(models["lgbm"], out / f"h{key}_lgbm.txt.gz"),
            "quantiles": {},
            "columns": models["cols"],
            "lgbm_params": {k: v for k, v in params.items() if not callable(v)},
            "cqr": {},
            "calibration_rows": len(cal),
        }
        for tau, m in models["quantiles"].items():
            info["quantiles"][f"{tau:g}"] = _save_lgbm(
                m, out / f"h{key}_q{round(tau * 100):02d}.txt.gz"
            )
        joblib.dump(models["ridge"], out / f"h{key}_ridge.joblib")
        info["ridge"] = {"file": f"h{key}_ridge.joblib", "alpha": models["ridge"].alpha}
        for lev, alpha in LEVELS.items():
            lo_c, hi_c = f"q{round(alpha / 2 * 100):02d}", f"q{round((1 - alpha / 2) * 100):02d}"
            _, _, (a_lo, a_hi) = cqr_log(cal, cal, lo_c, hi_c, alpha)
            info["cqr"][str(lev)] = [float(a_lo), float(a_hi)]
        manifest["horizons"][key] = info
        # Grouped permutation importance of the inner model (it decided the selection)
        imp = models["imp"]
        src_imp.append(imp.assign(horizon_h=h))
        gain_imp.append(models["lgbm"].importance().assign(horizon_h=h))
        log.info(
            "%g h: %d of %d features, CQR 80 %% %+.3f/%+.3f (%.0f s)",
            h,
            len(models["cols"]),
            len(feats),
            *info["cqr"]["80"],
            time.perf_counter() - t0,
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.concat(src_imp).to_csv(out / "source_importance.csv", index=False)
    pd.concat(gain_imp).to_csv(out / "feature_importance.csv", index=False)
    print(f"Operational model saved to {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat operational-model",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--variant", default="long")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--horizons", nargs="+", type=float)
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--rapid", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    train_operational(
        cfg,
        cfg.variant(args.variant),
        Path(args.processed),
        args.horizons,
        max_rounds=args.max_rounds,
        rapid=args.rapid,
    )

"""LightGBM hyperparameter tuning with Optuna, without touching any evaluation year.

Tuning is done once per horizon, with the inner split of the FIRST fold: models are trained
on the inner training years and scored on the inner validation years. All those years come
before any calibration or test year, so the chosen parameters can then be used in every
fold without leakage. (Tuning within each fold would be more precise, but would multiply
the run time by five.)

Criterion: mean squared error in m³/s on the inner validation (the same one used by the
source selection). The first trial is always the default configuration, so the result
cannot be worse than the starting point on the validation.

  uv run llobregat tune --variant long [--trials 30] [--horizons 1 6 24]
  uv run llobregat train --variant long --models lgbm --params tuned
  uv run llobregat tune --variant long --final    (tuned only where they improve > 5 %)

Output in data/processed/<station>/<variant>/tuning/: params_<h>h.json (the best),
trials_<h>h.csv (all trials) and summary.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import ModelVariant, StationConfig, load_station_config
from llobregat.experiment import (
    OWN_FLOW,
    dataset_dir,
    dataset_folds,
    feature_columns,
    load_dataset,
    target_column,
)
from llobregat.models.forecasters import LGBM_PARAMS, LGBMForecaster, mse

log = logging.getLogger("tune")

TUNED_KEYS = (
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l2",
    "min_gain_to_split",
    "max_depth",
)


def suggest(trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 10, 16]),
    }


def default_trial() -> dict:
    return {k: LGBM_PARAMS.get(k, -1 if k == "max_depth" else 0.0) for k in TUNED_KEYS}


def params_dir(processed: Path, cfg: StationConfig, variant: ModelVariant) -> Path:
    return dataset_dir(processed, cfg, variant) / "tuning"


def load_tuned(directory: Path, h: float) -> dict:
    f = Path(directory) / f"params_{h:g}h.json"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat tune first")
    return json.loads(f.read_text(encoding="utf-8"))["params"]


def tune(
    cfg: StationConfig,
    variant: ModelVariant,
    processed: Path,
    horizons: list[float],
    n_trials: int = 30,
    thin: int = 1,
    max_rounds: int = 3000,
    seed: int = 0,
    features: str = "selection",
) -> pd.DataFrame:
    import optuna

    from llobregat.train import selected_features

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, _ = load_dataset(processed, cfg, variant)
    folds = [f for f in dataset_folds(data, cfg, variant) if f.name != "final"]
    f0 = folds[0]
    feats = selected_features(processed, cfg, variant, feature_columns(data), features)
    q = data[OWN_FLOW]
    out = params_dir(processed, cfg, variant)
    out.mkdir(parents=True, exist_ok=True)
    log.info(
        "%s: tuning with inner training %s–%s and validation %s (%d features)",
        variant.name,
        f0.fold.train_years[0],
        f0.fold.inner_val_years[0] - 1,
        "–".join(map(str, f0.fold.inner_val_years)),
        len(feats),
    )
    summary = []
    for h in horizons:
        y = data[target_column(h)]
        base = (q.notna() & y.notna()).to_numpy()
        itr = np.flatnonzero(f0.inner_train & base)[::thin]
        iv = f0.inner_val & base
        X_tr, y_tr, q_tr = data.iloc[itr][feats], y.iloc[itr], q.iloc[itr]
        X_iv, y_iv, q_iv = data.loc[iv, feats], y[iv], q[iv]

        def objective(trial, X_tr=X_tr, y_tr=y_tr, q_tr=q_tr, X_iv=X_iv, y_iv=y_iv, q_iv=q_iv):
            params = {**LGBM_PARAMS, **suggest(trial)}
            m = LGBMForecaster(params=params, max_rounds=max_rounds)
            t0 = time.perf_counter()
            m.fit(X_tr, y_tr, q_tr, X_iv, y_iv, q_iv)
            trial.set_user_attr("iterations", m.best_iteration)
            trial.set_user_attr("seconds", round(time.perf_counter() - t0, 1))
            return mse(y_iv, m.predict(X_iv, q_iv))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.enqueue_trial(default_trial())
        study.optimize(objective, n_trials=n_trials)
        trials = study.trials_dataframe()
        trials.to_csv(out / f"trials_{h:g}h.csv", index=False, encoding="utf-8-sig")
        default_mse = study.trials[0].value
        best = study.best_trial
        (out / f"params_{h:g}h.json").write_text(
            json.dumps(
                {
                    "horizon_h": h,
                    "params": best.params,
                    "mse_validation": best.value,
                    "mse_validation_default": default_mse,
                    "iterations": best.user_attrs.get("iterations"),
                    "validation_years": [int(y) for y in f0.fold.inner_val_years],
                    "trials": n_trials,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        summary.append(
            {
                "horizon_h": h,
                "rmse_default": np.sqrt(default_mse),
                "rmse_tuned": np.sqrt(best.value),
                "gain_pct": 100 * (1 - np.sqrt(best.value / default_mse)),
                **best.params,
            }
        )
        log.info(
            "%g h: validation RMSE %.2f → %.2f m³/s (%s)",
            h,
            np.sqrt(default_mse),
            np.sqrt(best.value),
            best.params,
        )
    table = pd.DataFrame(summary)
    text = (
        f"=== Variant {variant.name}: hyperparameter tuning (inner validation "
        f"{', '.join(map(str, f0.fold.inner_val_years))}) ===\n"
        + table.round(4).to_string(index=False)
    )
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")
    return table


def final_params(directory: Path, horizons: list[float], min_gain_pct: float = 5.0) -> Path:
    """Final parameters per horizon, decided with the inner validation only.

    The tuned ones are used if they improve the validation RMSE by more than min_gain_pct %;
    otherwise, the defaults. With two validation years and few floods, a 1 % improvement is
    noise, and the tuned parameters tend to be more complex (more leaves, more depth):
    without a clear improvement, it is more prudent to keep the simple ones.
    Saves the result to <directory>/final/, which can be passed to train with --params.
    """
    out = Path(directory) / "final"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in horizons:
        f = Path(directory) / f"params_{h:g}h.json"
        info = json.loads(f.read_text(encoding="utf-8"))
        gain = 100 * (1 - np.sqrt(info["mse_validation"] / info["mse_validation_default"]))
        use_tuned = gain > min_gain_pct
        params = info["params"] if use_tuned else default_trial()
        (out / f"params_{h:g}h.json").write_text(
            json.dumps(
                {
                    "horizon_h": h,
                    "params": params,
                    "source": "tuned" if use_tuned else "default",
                    "validation_gain_pct": round(gain, 2),
                    "threshold_pct": min_gain_pct,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "horizon_h": h,
                "validation_gain_pct": round(gain, 2),
                "params": "tuned" if use_tuned else "default",
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(f"Saved to {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat tune",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--variant", required=True)
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--horizons", nargs="+", type=float)
    p.add_argument("--trials", type=int, default=30, help="Trials per horizon")
    p.add_argument("--thin", type=int, default=1, help="Use 1 in every N training rows")
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--features", default="selection", choices=["selection", "all"])
    p.add_argument(
        "--final",
        action="store_true",
        help="No tuning: choose per horizon between the tuned and default ones (validation)",
    )
    p.add_argument("--min-gain", type=float, default=5.0, help="With --final, in %%")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    v = cfg.variant(args.variant)
    if args.final:
        final_params(
            params_dir(Path(args.processed), cfg, v),
            args.horizons or cfg.horizons_h,
            args.min_gain,
        )
        return
    tune(
        cfg,
        v,
        Path(args.processed),
        args.horizons or cfg.horizons_h,
        args.trials,
        args.thin,
        args.max_rounds,
        features=args.features,
    )

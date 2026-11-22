"""Reference models and LightGBM, with validation by hydrological year.

For each outer fold (the 'final' hold-out is not touched: it is evaluated once, with the
chosen model) and each horizon:

  persistence    Q(t+h) = Q(t)
  climatology    median flow for the day of year of t+h (±7 days), from the training period
  ridge          L2 linear regression on the kept features, on the flow change Q(t+h) − Q(t);
                 the penalty is chosen on the inner validation
  lgbm           LightGBM. 1) Trained on the inner training period and stopped on the inner
                 validation (which gives the number of iterations). 2) Source selection by
                 grouped permutation importance on the inner validation (selection.py).
                 3) Retrained on the whole training period with the selected sources only.

The calibration period of each fold is not used here: it is kept for the prediction
intervals (probabilistic.py). All models are evaluated on exactly the same rows: those with
both the current and the future flow observed.

  uv run llobregat train --variant long [--models persistence climatology ridge lgbm]
        [--horizons 1 6 24] [--folds 2023 2024] [--features selection|all]
        [--no-selection] [--transform log_ratio|delta]
        [--weights uniform|tfg_stepped|power|episodes]
  uv run llobregat compare --variant long
  uv run llobregat compare --variant all     (+ comparison of variants on the shared years)

Output in data/processed/<station>/<variant>/models/<model>/: predictions.parquet,
metrics.csv, config.json and, for LightGBM, source_importance.csv and
feature_importance.csv. The comparison goes to models/comparison*.csv and models/summary.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat import weights as W
from llobregat.analysis import feature_groups
from llobregat.config import ModelVariant, StationConfig, load_station_config
from llobregat.experiment import (
    OWN_FLOW,
    FoldMasks,
    dataset_dir,
    dataset_folds,
    feature_columns,
    hydro_years_of,
    load_dataset,
    target_column,
)
from llobregat.metrics import (
    block_bootstrap_diff,
    continuous_scores,
    diebold_mariano,
    episode_table,
    event_scores,
    nse,
)
from llobregat.models.baselines import Climatology
from llobregat.models.forecasters import LGBM_PARAMS, LGBMForecaster, RidgeForecaster
from llobregat.selection import columns_of, group_permutation_importance, select_groups

log = logging.getLogger("train")

MODELS = ("persistence", "climatology", "ridge", "lgbm")
HIGH_FLOW_QUANTILE = 0.95


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def selected_features(
    processed: Path, cfg: StationConfig, variant: ModelVariant, feats: list[str], mode: str
) -> list[str]:
    if mode == "all":
        return feats
    f = dataset_dir(processed, cfg, variant) / "feature_analysis" / "feature_selection.csv"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat analyse-features first (or --features all)")
    sel = pd.read_csv(f, encoding="utf-8-sig")
    keep = set(sel.loc[sel["keep"] == 1, "feature"])
    return [c for c in feats if c in keep]


def choose_folds(folds: list[FoldMasks], wanted: list[int] | None) -> list[FoldMasks]:
    outer = [f for f in folds if f.name != "final"]
    if wanted:
        outer = [f for f in outer if any(y in wanted for y in f.fold.test_years)]
    if not outer:
        raise SystemExit("No fold matches --folds")
    return outer


def thresholds(q_train: pd.Series, cfg: StationConfig) -> dict[str, float]:
    """Thresholds of the fold, computed from the training period only."""
    return {
        "high": float(q_train.quantile(HIGH_FLOW_QUANTILE)),
        "warning": float(
            cfg.thresholds.warning_m3s or q_train.quantile(cfg.thresholds.warning_quantile)
        ),
    }


def model_name(model: str, args: argparse.Namespace) -> str:
    if model in ("persistence", "climatology"):
        return model
    parts = [model]
    if args.features == "all":
        parts.append("allfeatures")
    if model == "lgbm" and args.no_selection:
        parts.append("noselection")
    if model == "lgbm" and args.transform != "log_ratio":
        parts.append(args.transform)
        if args.transform == "level" and args.tweedie:
            parts.append("tweedie")
    if model == "ridge" and args.ridge_transform != "delta":
        parts.append(args.ridge_transform)
    if model == "lgbm" and args.weights != "uniform":
        parts.append(args.weights)
    if model == "lgbm" and getattr(args, "linear_trees", False):
        parts.append("linear")
    if model == "lgbm" and getattr(args, "params", None):
        parts.append("tuned")
    if args.name:
        parts.append(args.name)
    return "_".join(parts)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
WEIGHTS = ("uniform", "tfg_stepped", "power", "episodes")


def sample_weights(scheme: str, y, h: float, step_h: float) -> np.ndarray:
    """Training weights from the future flow Q(t+h) of the training rows.

    uniform      all equal
    tfg_stepped  those of the bachelor thesis: 3/5/15/40 above Q75/Q90/Q95/Q99, 100 in floods
    power        (Q/Q̄)^0.5: grows with the flow, but much more gently than the thesis scheme
    episodes     10 on the rows where Q(t+h) exceeds p95 and in the 6 h before (the rise)
    All are normalised to mean 1; the evaluation never uses weights.
    """
    y = np.asarray(y, dtype=float)
    if scheme == "uniform":
        return W.uniform(y)
    if scheme == "tfg_stepped":
        return W.tfg_stepped(y)
    if scheme == "power":
        return W.power(y, alpha=0.5)
    if scheme == "episodes":
        return W.event_window(
            y, thr=float(np.nanquantile(y, 0.95)), c=10.0, before_steps=round(6 / step_h)
        )
    raise ValueError(f"Unknown weighting scheme: {scheme}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(pred: pd.DataFrame, thr: dict[str, float], h: float, step_h: float) -> dict:
    """Metrics of one fold and horizon. Episodes are searched on the target-time axis
    (t + h), on a regular grid."""
    obs, p, qn = pred["obs"], pred["pred"], pred["q_now"]
    out = continuous_scores(obs, p, persistence=qn)
    hi = obs > thr["high"]
    if hi.any():
        err = p[hi] - obs[hi]
        out["rmse_high"] = float(np.sqrt(np.mean(err**2)))
        out["bias_high_pct"] = float(100 * err.mean() / obs[hi].mean())
        out["pi_high"] = float(1 - np.sum(err**2) / np.sum((qn[hi] - obs[hi]) ** 2))
    out.update(event_metrics(pred, thr["warning"], h, step_h))
    out["warning_threshold"] = thr["warning"]
    out["high_threshold"] = thr["high"]
    return out


def _episodes(
    pred: pd.DataFrame, threshold: float, h: float, step_h: float
) -> tuple[dict, pd.DataFrame]:
    t_target = pred.index + pd.Timedelta(hours=h)
    grid = pd.date_range(t_target.min(), t_target.max(), freq=pd.Timedelta(hours=step_h))
    o = pd.Series(pred["obs"].to_numpy(), index=t_target).reindex(grid)
    pp = pd.Series(pred["pred"].to_numpy(), index=t_target).reindex(grid)
    ev = event_scores(o.to_numpy(), (pp > threshold).to_numpy(), threshold)
    ep = pd.DataFrame()
    if ev["hits"] + ev["misses"]:
        window = round(max(12.0, h + 6.0) / step_h)
        ep = episode_table(o, pp, threshold, search_steps=window)
    return ev, ep


def _episode_summary(ev: dict, ep: pd.DataFrame) -> dict:
    h_, m_, f_ = ev["hits"], ev["misses"], ev["false_alarms"]
    out = {
        "pod": h_ / (h_ + m_) if h_ + m_ else np.nan,
        "far": f_ / (h_ + f_) if h_ + f_ else np.nan,
        "csi": h_ / (h_ + m_ + f_) if h_ + m_ + f_ else np.nan,
        "episodes": h_ + m_,
        "hits": h_,
        "false_alarms": f_,
        "peak_error_pct": np.nan,
        "peak_abs_error_pct": np.nan,
        "peak_timing_error_h": np.nan,
    }
    if len(ep):
        out["peak_error_pct"] = float(100 * ep["peak_rel_error"].median())
        out["peak_abs_error_pct"] = float(100 * ep["peak_rel_error"].abs().median())
        out["peak_timing_error_h"] = float(ep["peak_timing_error_h"].median())
    return out


def event_metrics(pred: pd.DataFrame, threshold: float, h: float, step_h: float) -> dict:
    """Warning skill (POD, FAR, CSI) and peak errors, on the target-time axis (t + h).

    The forecast peak is searched up to max(12 h, h + 6 h) from the observed peak: with a
    fixed 12 h window, at 24 h the persistence peak (which arrives 24 h late) fell outside
    and the peak and timing errors of that horizon were not comparable.
    """
    return _episode_summary(*_episodes(pred, threshold, h, step_h))


def pooled_metrics(pred: pd.DataFrame, met: pd.DataFrame, step_h: float) -> pd.DataFrame:
    """Metrics over all the test years of a model pooled together (not the mean of folds).

    The mean of the fold NSEs is misleading: a drought year has so little variance that its
    NSE can be very negative and drag the mean down, even if the errors in m³/s are small.
    The pooled NSE weighs each year by its variability. The high-flow threshold is the p95 of
    the observed test flows (the same for every model, since they are evaluated on the same
    rows); the warning threshold is that of each fold. Episodes are accumulated across folds.
    """
    rows = []
    for h, part in pred.groupby("horizon_h"):
        out = continuous_scores(part["obs"], part["pred"], persistence=part["q_now"])
        thr_hi = float(part["obs"].quantile(HIGH_FLOW_QUANTILE))
        hi = part["obs"] > thr_hi
        err = part.loc[hi, "pred"] - part.loc[hi, "obs"]
        out["rmse_high"] = float(np.sqrt(np.mean(err**2)))
        out["bias_high_pct"] = float(100 * err.mean() / part.loc[hi, "obs"].mean())
        out["pi_high"] = float(
            1 - np.sum(err**2) / np.sum((part.loc[hi, "q_now"] - part.loc[hi, "obs"]) ** 2)
        )
        tot = {"hits": 0, "misses": 0, "false_alarms": 0}
        eps = []
        for fold, sub in part.groupby("fold"):
            thr = met.loc[(met["fold"] == fold) & (met["horizon_h"] == h), "warning_threshold"]
            if thr.empty or not np.isfinite(thr.iloc[0]):
                continue
            ev, ep = _episodes(sub, float(thr.iloc[0]), h, step_h)
            for k in tot:
                tot[k] += ev[k]
            eps.append(ep)
        ep_all = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
        out.update(_episode_summary(tot, ep_all))
        out.update(horizon_h=h, rows=len(part))
        rows.append(out)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def lgbm_params(args: argparse.Namespace, h: float | None = None) -> dict:
    p = dict(LGBM_PARAMS)
    tuned = getattr(args, "params_dir", None)
    if tuned is not None and h is not None:
        from llobregat.tuning import load_tuned

        p.update(load_tuned(tuned, h))
    if args.rapid:
        p["learning_rate"] = 0.1
    if getattr(args, "linear_trees", False):
        # Each leaf fits a linear regression: the model can extrapolate a rise instead of
        # staying at the largest value seen in training
        p["linear_tree"] = True
        p["linear_lambda"] = 1.0
    if args.transform == "level" and args.tweedie:
        p["objective"] = "tweedie"
        p["tweedie_variance_power"] = 1.5
    return p


def run(cfg: StationConfig, variant: ModelVariant, processed: Path, args) -> dict[str, Path]:
    data, _ = load_dataset(processed, cfg, variant)
    folds = choose_folds(dataset_folds(data, cfg, variant), args.folds)
    feats = selected_features(processed, cfg, variant, feature_columns(data), args.features)
    groups = feature_groups(feats)
    horizons = args.horizons or cfg.horizons_h
    step_h = cfg.resolution_min / 60
    q = data[OWN_FLOW]
    years = hydro_years_of(data.index, cfg)
    names = {m: model_name(m, args) for m in args.models}
    acc = {m: {"pred": [], "met": [], "imp_g": [], "imp_v": []} for m in args.models}
    log.info(
        "%s: %d folds, horizons %s, %d features in %d sources",
        variant.name,
        len(folds),
        horizons,
        len(feats),
        len(groups),
    )

    for fm in folds:
        f = fm.fold
        thr = thresholds(q[fm.train].dropna(), cfg)
        clim = Climatology().fit(q[fm.train]) if "climatology" in args.models else None
        for h in horizons:
            y = data[target_column(h)]
            base = (q.notna() & y.notna()).to_numpy()
            tr, itr, iv, te = (
                getattr(fm, k) & base for k in ("train", "inner_train", "inner_val", "test")
            )
            if not te.any():
                log.warning("%s %g h: no test rows", f.name, h)
                continue
            idx_te = data.index[te]
            frame = pd.DataFrame({"obs": y[te].to_numpy(), "q_now": q[te].to_numpy()}, index=idx_te)
            for m in args.models:
                t0 = time.perf_counter()
                extra: dict = {}
                if m == "persistence":
                    pred = q[te].to_numpy()
                elif m == "climatology":
                    pred = clim.predict(
                        idx_te, h_steps=cfg.steps(h), resolution_min=cfg.resolution_min
                    )
                elif m == "ridge":
                    r = RidgeForecaster(transform=args.ridge_transform)
                    r.tune(data.loc[itr, feats], y[itr], q[itr], data.loc[iv, feats], y[iv], q[iv])
                    r.fit(data.loc[tr, feats], y[tr], q[tr])
                    pred = r.predict(data.loc[te, feats], q[te])
                    extra["ridge_alpha"] = r.alpha
                else:  # lgbm
                    g = LGBMForecaster(
                        transform=args.transform,
                        params=lgbm_params(args, h),
                        max_rounds=args.max_rounds,
                    )
                    g.fit(
                        data.loc[itr, feats],
                        y[itr],
                        q[itr],
                        data.loc[iv, feats],
                        y[iv],
                        q[iv],
                        weight=sample_weights(args.weights, y[itr], h, step_h),
                    )
                    cols = feats
                    if not args.no_selection:
                        X_iv, q_iv = data.loc[iv, feats], q[iv].to_numpy()
                        imp = group_permutation_importance(
                            lambda X, g=g, q_iv=q_iv: g.predict(X, q_iv), X_iv, y[iv], groups
                        )
                        kept = select_groups(imp)
                        cols = columns_of(kept, feats)
                        imp["kept"] = imp["group"].isin(kept).astype(int)
                        acc[m]["imp_g"].append(imp.assign(fold=f.name, horizon_h=h))
                    final = LGBMForecaster(transform=args.transform, params=lgbm_params(args, h))
                    final.fit(
                        data.loc[tr, cols],
                        y[tr],
                        q[tr],
                        weight=sample_weights(args.weights, y[tr], h, step_h),
                        rounds=g.best_iteration,
                    )
                    pred = final.predict(data.loc[te, cols], q[te])
                    acc[m]["imp_v"].append(final.importance().assign(fold=f.name, horizon_h=h))
                    extra.update(iterations=g.best_iteration, n_features=len(cols))
                p = frame.assign(pred=np.asarray(pred, dtype=float))
                met = evaluate(p, thr, h, step_h)
                met.update(
                    model=names[m],
                    fold=f.name,
                    test_year=",".join(map(str, f.test_years)),
                    horizon_h=h,
                    rows=len(p),
                    seconds=round(time.perf_counter() - t0, 1),
                    **extra,
                )
                acc[m]["met"].append(met)
                acc[m]["pred"].append(
                    p.assign(fold=f.name, horizon_h=h, hydro_year=years[te]).astype(
                        {"obs": "float32", "q_now": "float32", "pred": "float32"}
                    )
                )
                log.info(
                    "%-8s %5gh %-22s NSE %.3f  PI %+.3f  CSI %s  (%.0f s)",
                    f.name,
                    h,
                    names[m],
                    met["nse"],
                    met["pi"],
                    f"{met['csi']:.2f}" if np.isfinite(met["csi"]) else "—",
                    met["seconds"],
                )

    root = dataset_dir(processed, cfg, variant) / "models"
    written = {}
    for m in args.models:
        out = root / names[m]
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(acc[m]["pred"]).to_parquet(out / "predictions.parquet")
        pd.DataFrame(acc[m]["met"]).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
        if acc[m]["imp_g"]:
            ig = pd.concat(acc[m]["imp_g"], ignore_index=True)
            ig.to_csv(out / "source_importance.csv", index=False, encoding="utf-8-sig")
        if acc[m]["imp_v"]:
            iv_ = pd.concat(acc[m]["imp_v"], ignore_index=True)
            iv_.to_csv(out / "feature_importance.csv", index=False, encoding="utf-8-sig")
        cfg_run = {
            "model": names[m],
            "variant": variant.name,
            "folds": [fm.name for fm in folds],
            "test_years": [[int(y) for y in fm.fold.test_years] for fm in folds],
            "horizons_h": horizons,
            "features": args.features,
            "n_features": len(feats),
            "transform": (args.ridge_transform if m == "ridge" else args.transform),
            "source_selection": m == "lgbm" and not args.no_selection,
            "weights": args.weights if m == "lgbm" else None,
            "lgbm_params": (
                {f"{h:g}h": lgbm_params(args, h) for h in horizons} if m == "lgbm" else None
            ),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (out / "config.json").write_text(json.dumps(cfg_run, ensure_ascii=False, indent=2))
        written[names[m]] = out
    summary = pd.concat([pd.DataFrame(acc[m]["met"]) for m in args.models], ignore_index=True)
    print_summary(summary, variant.name)
    print(f"Saved to {root}")
    return written


def print_summary(met: pd.DataFrame, variant: str) -> None:
    pd.set_option("display.width", 200)
    print(f"\n=== Variant {variant}: mean of the folds ===")
    for col, title in (
        ("nse", "NSE"),
        ("pi", "Persistence index (PI > 0: better than persistence)"),
        ("rmse_high", "RMSE at high flows (> p95), m³/s"),
        ("csi", "Warning CSI (p99 threshold)"),
        ("peak_error_pct", "Peak error (median, %; negative = undershoot)"),
    ):
        if col not in met:
            continue
        t = met.pivot_table(index="model", columns="horizon_h", values=col, aggfunc="mean")
        t.columns = [f"{c:g} h" for c in t.columns]
        print(f"\n{title}:")
        print(t.round(3).to_string())


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
BASELINES = ("persistence", "climatology")


def refresh_event_metrics(
    met: pd.DataFrame, preds: dict[str, pd.DataFrame], step_h: float
) -> pd.DataFrame:
    """Recompute the episode metrics from the saved predictions (and the warning threshold
    of each fold), so that every model is measured with the same rule even if it was trained
    with a different version of the code."""
    met = met.copy()
    for i, r in met.iterrows():
        p = preds.get(r["model"])
        if p is None or not np.isfinite(r.get("warning_threshold", np.nan)):
            continue
        sub = p[(p["fold"] == r["fold"]) & (p["horizon_h"] == r["horizon_h"])]
        if len(sub):
            for k, v in event_metrics(sub, r["warning_threshold"], r["horizon_h"], step_h).items():
                met.loc[i, k] = v
    return met


COMPARE_COLS = [
    "nse",
    "kge",
    "pi",
    "rmse",
    "rmse_high",
    "pi_high",
    "bias_high_pct",
    "pod",
    "far",
    "csi",
    "peak_error_pct",
    "peak_abs_error_pct",
    "peak_timing_error_h",
]


def compare(
    cfg: StationConfig,
    variant: ModelVariant,
    processed: Path,
    models: list[str] | None = None,
    references: tuple[str, ...] = ("persistence", "ridge"),
    n_boot: int = 500,
) -> pd.DataFrame:
    root = dataset_dir(processed, cfg, variant) / "models"
    dirs = sorted(d for d in root.glob("*") if (d / "metrics.csv").exists())
    if models:
        dirs = [d for d in dirs if d.name in models]
    if not dirs:
        raise SystemExit(f"No model in {root}: run llobregat train first")
    met = pd.concat([pd.read_csv(d / "metrics.csv") for d in dirs], ignore_index=True)
    preds = {d.name: pd.read_parquet(d / "predictions.parquet") for d in dirs}
    step_h = cfg.resolution_min / 60
    met = refresh_event_metrics(met, preds, step_h)
    met.to_csv(root / "metrics_all.csv", index=False, encoding="utf-8-sig")
    cols = [c for c in COMPARE_COLS if c in met]
    per_fold = met.groupby(["model", "horizon_h"])[cols].mean().round(4).reset_index()
    per_fold.to_csv(root / "comparison_fold_mean.csv", index=False, encoding="utf-8-sig")
    summary = pd.concat(
        [pooled_metrics(preds[m], met[met["model"] == m], step_h).assign(model=m) for m in preds],
        ignore_index=True,
    )
    summary.round(4).to_csv(root / "comparison.csv", index=False, encoding="utf-8-sig")

    rows = []
    for ref in references:
        if ref not in preds:
            continue
        for m, pm in preds.items():
            if m == ref or (ref != "persistence" and m in BASELINES):
                continue
            for h in sorted(pm["horizon_h"].unique()):
                a = pm[pm["horizon_h"] == h]
                b = preds[ref][preds[ref]["horizon_h"] == h]
                common = a.index.intersection(b.index)
                if len(common) < 100:
                    continue
                a, b = a.loc[common], b.loc[common]
                groups = common.strftime("%Y-%m")  # one-month blocks: respect the dependence
                bs = block_bootstrap_diff(
                    nse, a["obs"], a["pred"], b["pred"], groups, n_boot=n_boot
                )
                ea, eb = (a["pred"] - a["obs"]) ** 2, (b["pred"] - b["obs"]) ** 2
                dm = diebold_mariano(ea, eb, h=max(1, round(h / step_h)))
                rows.append(
                    {
                        "model": m,
                        "reference": ref,
                        "horizon_h": h,
                        "rows": len(common),
                        "nse_model": nse(a["obs"], a["pred"]),
                        "nse_reference": nse(b["obs"], b["pred"]),
                        "nse_diff": bs["diff"],
                        "nse_diff_ci95_low": bs["ci_low"],
                        "nse_diff_ci95_high": bs["ci_high"],
                        "dm_stat": dm["dm_stat"],
                        "dm_p": dm["p_value"],
                    }
                )
    stats = pd.DataFrame(rows)
    stats.round(4).to_csv(root / "comparison_stats.csv", index=False, encoding="utf-8-sig")

    years = sorted({str(y) for p_ in preds.values() for y in p_["hydro_year"].unique()})
    lines = [
        f"=== Variant {variant.name}: model comparison (test years pooled: {', '.join(years)}) ==="
    ]
    for col, title in (
        ("nse", "NSE"),
        ("pi", "Persistence index"),
        ("rmse_high", "RMSE at high flows (> p95), m³/s"),
        ("csi", "Warning CSI"),
        ("far", "False alarm ratio (FAR)"),
        ("peak_error_pct", "Peak error, median (%)"),
        ("peak_timing_error_h", "Peak timing error, median (h; negative = early)"),
    ):
        if col in summary:
            t = summary.pivot(index="model", columns="horizon_h", values=col)
            t.columns = [f"{c:g} h" for c in t.columns]
            lines += ["", f"{title}:", t.round(3).to_string()]
    if len(stats):
        lines += ["", "NSE difference with the reference (monthly block bootstrap, 95 % CI):"]
        for r in stats.itertuples():
            sig = "*" if r.nse_diff_ci95_low > 0 or r.nse_diff_ci95_high < 0 else " "
            lines.append(
                f"  {r.model:<24} vs {r.reference:<13} {r.horizon_h:>4g} h  "
                f"{r.nse_diff:+.3f} [{r.nse_diff_ci95_low:+.3f}, {r.nse_diff_ci95_high:+.3f}]{sig}"
                f"  DM p={r.dm_p:.3g}"
            )
        lines.append("  * the interval does not include 0")
    text = "\n".join(lines)
    (root / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {root}")
    return summary


def compare_variants(
    cfg: StationConfig, processed: Path, models: list[str] | None = None, n_boot: int = 500
) -> pd.DataFrame:
    """Compare the variants on the test years they share only (e.g. the long variant has
    more folds than the full one: the means over all folds are not comparable).
    """
    step_h = cfg.resolution_min / 60
    met_all, preds_all = [], {}
    for v in cfg.variants:
        root = dataset_dir(processed, cfg, v) / "models"
        for d in sorted(root.glob("*")):
            if not (d / "metrics.csv").exists() or (models and d.name not in models):
                continue
            pr = pd.read_parquet(d / "predictions.parquet")
            m = refresh_event_metrics(pd.read_csv(d / "metrics.csv"), {d.name: pr}, step_h)
            met_all.append(m.assign(variant=v.name))
            preds_all[(v.name, d.name)] = pr
    if not met_all:
        raise SystemExit("No trained models")
    met = pd.concat(met_all, ignore_index=True)
    years = [set(met.loc[met["variant"] == v.name, "test_year"].astype(str)) for v in cfg.variants]
    years = [y for y in years if y]
    common = sorted(set.intersection(*years)) if len(years) > 1 else []
    if not common:
        raise SystemExit("The variants share no test year")
    parts = []
    for (vn, mn), pr in preds_all.items():
        pr_c = pr[pr["hydro_year"].astype(str).isin(common)]
        m_c = met[(met["variant"] == vn) & (met["model"] == mn)]
        if len(pr_c):
            parts.append(pooled_metrics(pr_c, m_c, step_h).assign(variant=vn, model=mn))
    summary = pd.concat(parts, ignore_index=True).round(4)
    out = Path(processed) / cfg.station_id
    summary.to_csv(out / "variant_comparison.csv", index=False, encoding="utf-8-sig")

    names = sorted({m for (_, m) in preds_all} - set(BASELINES))
    vnames = [v.name for v in cfg.variants]
    rows = []
    for m in names:
        for i, a_v in enumerate(vnames):
            for b_v in vnames[i + 1 :]:
                if (a_v, m) not in preds_all or (b_v, m) not in preds_all:
                    continue
                A, B = preds_all[(a_v, m)], preds_all[(b_v, m)]
                for h in sorted(set(A["horizon_h"]) & set(B["horizon_h"])):
                    a = A[(A["horizon_h"] == h) & A["hydro_year"].astype(str).isin(common)]
                    b = B[(B["horizon_h"] == h) & B["hydro_year"].astype(str).isin(common)]
                    idx = a.index.intersection(b.index)
                    if len(idx) < 100:
                        continue
                    a, b = a.loc[idx], b.loc[idx]
                    bs = block_bootstrap_diff(
                        nse, a["obs"], a["pred"], b["pred"], idx.strftime("%Y-%m"), n_boot=n_boot
                    )
                    rows.append(
                        {
                            "model": m,
                            "variant_a": a_v,
                            "variant_b": b_v,
                            "horizon_h": h,
                            "rows": len(idx),
                            "nse_a": nse(a["obs"], a["pred"]),
                            "nse_b": nse(b["obs"], b["pred"]),
                            "nse_diff": bs["diff"],
                            "nse_diff_ci95_low": bs["ci_low"],
                            "nse_diff_ci95_high": bs["ci_high"],
                        }
                    )
    stats = pd.DataFrame(
        rows,
        columns=[
            "model",
            "variant_a",
            "variant_b",
            "horizon_h",
            "rows",
            "nse_a",
            "nse_b",
            "nse_diff",
            "nse_diff_ci95_low",
            "nse_diff_ci95_high",
        ],
    )
    stats.round(4).to_csv(out / "variant_comparison_stats.csv", index=False, encoding="utf-8-sig")
    lines = [f"=== Variants on the same test years ({', '.join(common)}) ==="]
    summary["model_variant"] = summary["model"] + " (" + summary["variant"] + ")"
    for col, title in (
        ("nse", "NSE"),
        ("pi", "Persistence index"),
        ("rmse_high", "RMSE at high flows (> p95), m³/s"),
        ("csi", "Warning CSI"),
        ("peak_error_pct", "Peak error, median (%)"),
    ):
        if col in summary:
            tb = summary.pivot(index="model_variant", columns="horizon_h", values=col)
            tb.columns = [f"{c:g} h" for c in tb.columns]
            lines += ["", f"{title}:", tb.round(3).to_string()]
    if len(stats):
        lines += ["", "NSE difference between variants (monthly block bootstrap, 95 % CI):"]
        for r in stats.itertuples():
            sig = "*" if r.nse_diff_ci95_low > 0 or r.nse_diff_ci95_high < 0 else " "
            lines.append(
                f"  {r.model:<22} {r.variant_a} − {r.variant_b} {r.horizon_h:>4g} h  "
                f"{r.nse_diff:+.3f} [{r.nse_diff_ci95_low:+.3f}, {r.nse_diff_ci95_high:+.3f}]{sig}"
            )
        lines.append("  * the interval does not include 0")
    text = "\n".join(lines)
    (out / "variant_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")
    return summary


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--variant", required=True, help="long, full or all")
    p.add_argument("--processed", default="data/processed")


def _variants(cfg: StationConfig, name: str) -> list[ModelVariant]:
    return cfg.variants if name == "all" else [cfg.variant(name)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _common(p)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--horizons", nargs="+", type=float)
    p.add_argument("--folds", nargs="+", type=int, help="Test (hydrological) years")
    p.add_argument("--features", default="selection", choices=["selection", "all"])
    p.add_argument("--no-selection", action="store_true", help="LightGBM without source selection")
    p.add_argument(
        "--transform", default="log_ratio", choices=["log_ratio", "delta", "level", "log_level"]
    )
    p.add_argument("--tweedie", action="store_true", help="With --transform level")
    p.add_argument(
        "--ridge-transform",
        default="delta",
        choices=["log_ratio", "delta", "level", "log_level"],
        help="The linear model works better with the absolute change: on a log scale, a "
        "linear extrapolation blows up the peaks",
    )
    p.add_argument("--weights", default="uniform", choices=list(WEIGHTS))
    p.add_argument(
        "--linear-trees", action="store_true", help="LightGBM with linear regression in the leaves"
    )
    p.add_argument(
        "--params",
        help="'tuned' (those of llobregat tune) or a folder with params_<h>h.json",
    )
    p.add_argument("--rapid", action="store_true", help="Learning rate 0.1 (quick tests)")
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--name", help="Suffix of the model name")
    args = p.parse_args(argv)
    _logging()
    cfg = load_station_config(args.config)
    for v in _variants(cfg, args.variant):
        args.params_dir = None
        if args.params:
            from llobregat.tuning import params_dir

            args.params_dir = (
                params_dir(Path(args.processed), cfg, v)
                if args.params == "tuned"
                else Path(args.params)
            )
        run(cfg, v, Path(args.processed), args)


def main_compare(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="llobregat compare", description=compare.__doc__)
    _common(p)
    p.add_argument("--models", nargs="+", help="By default, all trained models")
    p.add_argument(
        "--references",
        nargs="+",
        default=["persistence", "ridge"],
        help="Models the others are compared with (e.g. persistence ridge lgbm)",
    )
    p.add_argument("--bootstrap", type=int, default=500)
    args = p.parse_args(argv)
    _logging()
    cfg = load_station_config(args.config)
    for v in _variants(cfg, args.variant):
        compare(cfg, v, Path(args.processed), args.models, tuple(args.references), args.bootstrap)
    if args.variant == "all" and len(cfg.variants) > 1:
        compare_variants(cfg, Path(args.processed), args.models, args.bootstrap)


def combine(
    cfg: StationConfig,
    variant: ModelVariant,
    processed: Path,
    models: list[str],
    weights: list[float] | None = None,
    name: str | None = None,
) -> Path:
    """Combination of trained models: weighted mean of their predictions.

    The weights must be fixed before looking at the test (equal by default): choosing them
    with the test results would be cheating. The rows are those common to all the models.
    """
    root = dataset_dir(processed, cfg, variant) / "models"
    w = np.asarray(weights or [1.0] * len(models), dtype=float)
    if len(w) != len(models):
        raise SystemExit("One weight per model is needed")
    w = w / w.sum()
    preds, mets = [], []
    for m in models:
        d = root / m
        if not (d / "predictions.parquet").exists():
            raise SystemExit(f"Missing {d}: train the model {m} first")
        preds.append(pd.read_parquet(d / "predictions.parquet"))
        mets.append(pd.read_csv(d / "metrics.csv"))
    key = ["horizon_h"]
    base = preds[0].set_index(key, append=True)
    combo = base["pred"].astype(float) * w[0]
    for wi, pr in zip(w[1:], preds[1:], strict=True):
        other = pr.set_index(key, append=True)["pred"].astype(float)
        combo = (combo + wi * other).dropna()
    out_pred = base.loc[combo.index].copy()
    out_pred["pred"] = combo.astype("float32")
    out_pred = out_pred.reset_index(level="horizon_h")
    name = name or "+".join(models)
    step_h = cfg.resolution_min / 60
    met0 = mets[0]
    rows = []
    for (fold, h), part in out_pred.groupby(["fold", "horizon_h"]):
        ref = met0[(met0["fold"] == fold) & (met0["horizon_h"] == h)].iloc[0]
        high = ref.get("high_threshold", np.nan)
        if not np.isfinite(high):
            high = float(part["obs"].quantile(HIGH_FLOW_QUANTILE))
        met = evaluate(part, {"high": high, "warning": float(ref["warning_threshold"])}, h, step_h)
        met.update(
            model=name,
            fold=fold,
            test_year=ref["test_year"],
            horizon_h=h,
            rows=len(part),
            seconds=0.0,
        )
        rows.append(met)
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    out_pred.to_parquet(out / "predictions.parquet")
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    (out / "config.json").write_text(
        json.dumps(
            {"model": name, "combination": dict(zip(models, w.round(4).tolist(), strict=True))},
            ensure_ascii=False,
            indent=2,
        )
    )
    log.info("Combination %s saved to %s", name, out)
    return out


def main_combine(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="llobregat combine", description=combine.__doc__)
    _common(p)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, help="Equal by default")
    p.add_argument("--name", help="Name of the combined model (default: model1+model2)")
    args = p.parse_args(argv)
    _logging()
    cfg = load_station_config(args.config)
    for v in _variants(cfg, args.variant):
        combine(cfg, v, Path(args.processed), args.models, args.weights, args.name)


def _logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

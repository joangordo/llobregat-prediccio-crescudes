"""Probabilistic forecasts (calibrated intervals) and warnings.

For each fold and horizon:

  1. Point forecast: equal-weight mean of LightGBM (relative change, source selection,
     parameters from `tuning/final` if they exist) and Ridge.
  2. Quantiles 0.05 / 0.10 / 0.90 / 0.95 with LightGBM (quantile loss on the same
     transformed target: quantiles are preserved by a monotonic transform) and the same
     features as the point LightGBM. They are sorted so that they do not cross.
  3. Conformal calibration (asymmetric CQR) on the fold's CALIBRATION YEAR, which no model
     has seen. It is done on a log scale: the adjustment is multiplicative, which suits
     errors that grow with the flow (a fixed ±5 m³/s would be huge at low flow and
     ridiculous in a flood).
  4. ACI (Gibbs and Candès, 2021) on top of CQR, on the test: the level adjusts itself if
     the recent coverage drifts. It respects the real delay: the error of a forecast issued
     at t is only known at t + h.
  5. Reference: conformal interval around persistence (quantiles of the relative changes
     observed in the calibration year). A probabilistic model should be narrower than this
     at equal coverage.
  6. Warnings when the fold threshold (training p99) is exceeded, with three rules: point
     forecast, upper bound of the 80 % interval (≈ P > 10 %) and of the 90 % interval
     (≈ P > 5 %). The rule is chosen with the calibration years (of the fold and the
     previous ones), never with the test.

  uv run llobregat intervals --variant long [--horizons 1 6 24] [--folds 2023 2024]
  uv run llobregat final-evaluation --variant long        (ONCE ONLY: hold-out)

Output in data/processed/<station>/<variant>/uncertainty/ (or final/): predictions.parquet,
metrics.csv, warnings.csv and summary.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
from llobregat.metrics import continuous_scores, coverage, interval_score
from llobregat.models.forecasters import LGBM_PARAMS, LGBMForecaster, RidgeForecaster
from llobregat.selection import columns_of, group_permutation_importance, select_groups

log = logging.getLogger("uncertainty")

EPS = 0.1
QUANTILES = (0.05, 0.10, 0.90, 0.95)
LEVELS = {80: 0.20, 90: 0.10}  # nominal level → alpha
RULES = ("central", "upper80", "upper90")
DEFAULT_RULE = "upper80"


# --------------------------------------------------------------------------- #
# Log scale
# --------------------------------------------------------------------------- #
def to_log(x) -> np.ndarray:
    return np.log(np.clip(np.asarray(x, dtype=float), 0, None) + EPS)


def from_log(z) -> np.ndarray:
    return np.clip(np.exp(np.asarray(z, dtype=float)) - EPS, 0, None)


def _q_index(n: int, level: float) -> int | None:
    """Position of the conformal quantile ⌈(n+1)·level⌉ (None: the interval is infinite)."""
    k = math.ceil((n + 1) * level)
    return None if k > n else k - 1


def cqr_log(cal: pd.DataFrame, df: pd.DataFrame, lo: str, hi: str, alpha: float):
    """Asymmetric CQR on a log scale: each bound is corrected with its own quantile."""
    y, l_, h_ = to_log(cal["obs"]), to_log(cal[lo]), to_log(cal[hi])
    ok = np.isfinite(y) & np.isfinite(l_) & np.isfinite(h_)
    s_lo, s_hi = np.sort((l_ - y)[ok]), np.sort((y - h_)[ok])
    k = _q_index(len(s_lo), 1 - alpha / 2)
    a_lo = np.inf if k is None else s_lo[k]
    a_hi = np.inf if k is None else s_hi[k]
    return from_log(to_log(df[lo]) - a_lo), from_log(to_log(df[hi]) + a_hi), (a_lo, a_hi)


def aci_log(
    cal: pd.DataFrame,
    df: pd.DataFrame,
    lo: str,
    hi: str,
    alpha: float,
    gamma: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACI on symmetric CQR scores on a log scale, with delay h.

    At each time t the interval [lo − q_t, hi + q_t] is issued, with q_t the 1 − α_t quantile
    of the calibration scores. When the observation arrives (t + h): α ← α + γ·(α − error).
    """
    y_c, l_c, h_c = to_log(cal["obs"]), to_log(cal[lo]), to_log(cal[hi])
    sc = np.maximum(l_c - y_c, y_c - h_c)
    sc = np.sort(sc[np.isfinite(sc)])
    n = len(sc)
    t = df.index.to_numpy()
    arrival = (df.index + pd.Timedelta(hours=h)).to_numpy()
    y, lo_l, hi_l = to_log(df["obs"]), to_log(df[lo]), to_log(df[hi])
    out_lo, out_hi, alphas = np.empty(len(df)), np.empty(len(df)), np.empty(len(df))
    a_t, j = alpha, 0
    for i in range(len(df)):
        while j < i and arrival[j] <= t[i]:  # observations that have already arrived
            if np.isfinite(y[j]):
                err = float(not (out_lo[j] <= y[j] <= out_hi[j]))
                a_t += gamma * (alpha - err)
            j += 1
        a = min(max(a_t, 1e-4), 1 - 1e-4)
        k = _q_index(n, 1 - a)
        q = sc[-1] if k is None else sc[k]
        out_lo[i], out_hi[i], alphas[i] = lo_l[i] - q, hi_l[i] + q, a_t
    return from_log(out_lo), from_log(out_hi), alphas


def persistence_conformal(cal: pd.DataFrame, df: pd.DataFrame, alpha: float):
    """Reference interval: Q(t) × empirical quantiles of Q(t+h)/Q(t) in the calibration year."""
    r = to_log(cal["obs"]) - to_log(cal["q_now"])
    r = np.sort(r[np.isfinite(r)])
    n = len(r)
    k_hi = _q_index(n, 1 - alpha / 2)
    k_lo = math.floor((n + 1) * (alpha / 2)) - 1
    r_lo = -np.inf if k_lo < 0 else r[k_lo]
    r_hi = np.inf if k_hi is None else r[k_hi]
    base = to_log(df["q_now"])
    return from_log(base + r_lo), from_log(base + r_hi)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def lgbm_params_for(processed: Path, cfg: StationConfig, variant: ModelVariant, h: float) -> dict:
    """LightGBM parameters: those in tuning/final if they exist; the defaults otherwise."""
    p = dict(LGBM_PARAMS)
    f = dataset_dir(processed, cfg, variant) / "tuning" / "final" / f"params_{h:g}h.json"
    if f.exists():
        p.update(json.loads(f.read_text(encoding="utf-8"))["params"])
    return p


def fit_models(
    data: pd.DataFrame,
    fm: FoldMasks,
    feats: list[str],
    h: float,
    params: dict,
    quantiles: tuple[float, ...],
    max_rounds: int,
):
    """Train the point forecast (LightGBM + Ridge) and the quantiles of one fold and horizon."""
    y = data[target_column(h)]
    q = data[OWN_FLOW]
    base = (q.notna() & y.notna()).to_numpy()
    tr, itr, iv = (getattr(fm, k) & base for k in ("train", "inner_train", "inner_val"))
    X = data[feats]
    # Point LightGBM with source selection (as in train.py)
    g = LGBMForecaster(params=params, max_rounds=max_rounds)
    g.fit(X[itr], y[itr], q[itr], X[iv], y[iv], q[iv])
    q_iv = q[iv].to_numpy()
    imp = group_permutation_importance(
        lambda Z: g.predict(Z, q_iv), X[iv], y[iv], feature_groups(feats)
    )
    cols = columns_of(select_groups(imp), feats)
    lgbm = LGBMForecaster(params=params).fit(X.loc[tr, cols], y[tr], q[tr], rounds=g.best_iteration)
    ridge = RidgeForecaster(transform="delta")
    ridge.tune(X[itr], y[itr], q[itr], X[iv], y[iv], q[iv]).fit(X[tr], y[tr], q[tr])
    qm = {}
    for tau in quantiles:
        pq = {**params, "objective": "quantile", "alpha": tau, "metric": "quantile"}
        m0 = LGBMForecaster(params=pq, max_rounds=max_rounds)
        m0.fit(X.loc[itr, cols], y[itr], q[itr], X.loc[iv, cols], y[iv], q[iv])
        qm[tau] = LGBMForecaster(params=pq).fit(
            X.loc[tr, cols], y[tr], q[tr], rounds=m0.best_iteration
        )
    return {"lgbm": lgbm, "ridge": ridge, "quantiles": qm, "cols": cols, "feats": feats, "imp": imp}


def predict_frame(models: dict, data: pd.DataFrame, mask: np.ndarray, h: float) -> pd.DataFrame:
    y, q = data[target_column(h)], data[OWN_FLOW]
    X = data.loc[mask, models["feats"]]
    qn = q[mask].to_numpy()
    df = pd.DataFrame({"obs": y[mask].to_numpy(), "q_now": qn}, index=data.index[mask])
    df["lgbm"] = models["lgbm"].predict(X[models["cols"]], qn)
    df["ridge"] = models["ridge"].predict(X, qn)
    df["central"] = (df["lgbm"] + df["ridge"]) / 2
    taus = sorted(models["quantiles"])
    raw = [models["quantiles"][t].predict(X[models["cols"]], qn) for t in taus]
    for t, v in zip(taus, np.sort(np.vstack(raw), axis=0), strict=True):  # no crossing
        df[f"q{round(t * 100):02d}"] = v
    return df


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def interval_metrics(df: pd.DataFrame, lo, hi, alpha: float, thr_hi: float) -> dict:
    obs = df["obs"].to_numpy()
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    high = obs > thr_hi
    rel = (hi - lo) / np.maximum(obs, EPS)
    return {
        "coverage": coverage(obs, lo, hi),
        "coverage_high": coverage(obs[high], lo[high], hi[high]) if high.any() else np.nan,
        "below_pct": float(100 * np.mean(obs < lo)),
        "above_pct": float(100 * np.mean(obs > hi)),
        "width_m3s": float(np.mean(hi - lo)),
        "median_rel_width": float(np.median(rel)),
        "width_high_m3s": float(np.mean((hi - lo)[high])) if high.any() else np.nan,
        "interval_score": interval_score(obs, lo, hi, alpha),
    }


def weighted_interval_score(obs, median, intervals: dict[float, tuple]) -> float:
    """WIS (Bracher et al., 2021): combines the error of the median and the intervals."""
    obs = np.asarray(obs, dtype=float)
    total = 0.5 * np.abs(obs - np.asarray(median, dtype=float))
    for alpha, (lo, hi) in intervals.items():
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        is_ = (hi - lo) + (2 / alpha) * ((lo - obs) * (obs < lo) + (obs - hi) * (obs > hi))
        total = total + (alpha / 2) * is_
    return float(np.nanmean(total / (len(intervals) + 0.5)))


def alert_counts(df: pd.DataFrame, value: str, threshold: float, h: float, step_h: float):
    from llobregat.train import _episodes

    frame = df[["obs"]].assign(pred=df[value].to_numpy())
    ev, ep = _episodes(frame, threshold, h, step_h)
    return ev, ep


def _scores(ev: dict) -> dict:
    h_, m_, f_ = ev["hits"], ev["misses"], ev["false_alarms"]
    return {
        "hits": h_,
        "misses": m_,
        "false_alarms": f_,
        "pod": h_ / (h_ + m_) if h_ + m_ else np.nan,
        "far": f_ / (h_ + f_) if h_ + f_ else np.nan,
        "csi": h_ / (h_ + m_ + f_) if h_ + m_ + f_ else np.nan,
    }


def choose_rule(cal_history: list[tuple[pd.DataFrame, float]], h: float, step_h: float) -> str:
    """Warning rule with the best CSI on the available calibration years (never the test)."""
    best, best_csi = DEFAULT_RULE, -1.0
    for rule in RULES:
        tot = {"hits": 0, "misses": 0, "false_alarms": 0}
        for cal, thr in cal_history:
            ev, _ = alert_counts(cal, rule, thr, h, step_h)
            for k in tot:
                tot[k] += ev[k]
        if tot["hits"] + tot["misses"] == 0:
            continue
        csi = _scores(tot)["csi"]
        if np.isfinite(csi) and csi > best_csi:
            best, best_csi = rule, csi
    return best


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def add_intervals(cal: pd.DataFrame, df: pd.DataFrame, h: float, gamma: float) -> dict:
    """Add to df (and to cal, the CQR ones) the bounds of each method and level."""
    adj = {}
    for lev, alpha in LEVELS.items():
        lo_c, hi_c = (f"q{round(alpha / 2 * 100):02d}", f"q{round((1 - alpha / 2) * 100):02d}")
        df[f"lo{lev}_quantiles"], df[f"hi{lev}_quantiles"] = df[lo_c], df[hi_c]
        lo, hi, a = cqr_log(cal, df, lo_c, hi_c, alpha)
        df[f"lo{lev}_cqr"], df[f"hi{lev}_cqr"] = lo, hi
        lo_k, hi_k, _ = cqr_log(cal, cal, lo_c, hi_c, alpha)
        cal[f"lo{lev}_cqr"], cal[f"hi{lev}_cqr"] = lo_k, hi_k
        adj[lev] = a
        lo, hi, alphas = aci_log(cal, df, lo_c, hi_c, alpha, gamma, h)
        df[f"lo{lev}_aci"], df[f"hi{lev}_aci"] = lo, hi
        df[f"alpha{lev}_aci"] = alphas
        lo, hi = persistence_conformal(cal, df, alpha)
        df[f"lo{lev}_persistence"], df[f"hi{lev}_persistence"] = lo, hi
    for d in (df, cal):
        d["upper80"], d["upper90"] = d["hi80_cqr"], d["hi90_cqr"]
    return adj


METHODS = ("quantiles", "cqr", "aci", "persistence")


def run(
    cfg: StationConfig,
    variant: ModelVariant,
    processed: Path,
    folds: list[FoldMasks],
    horizons: list[float],
    out: Path,
    quantiles: tuple[float, ...] = QUANTILES,
    max_rounds: int = 3000,
    rapid: bool = False,
) -> dict[str, pd.DataFrame]:
    from llobregat.train import selected_features, thresholds

    data, _ = load_dataset(processed, cfg, variant)
    feats = selected_features(processed, cfg, variant, feature_columns(data), "selection")
    q = data[OWN_FLOW]
    years = hydro_years_of(data.index, cfg)
    step_h = cfg.resolution_min / 60
    gamma = cfg.intervals.aci_gamma
    preds, mets, alerts = [], [], []
    cal_history: dict[float, list] = {h: [] for h in horizons}
    for fm in folds:
        f = fm.fold
        thr = thresholds(q[fm.train].dropna(), cfg)
        for h in horizons:
            t0 = time.perf_counter()
            params = lgbm_params_for(processed, cfg, variant, h)
            if rapid:
                params["learning_rate"] = 0.1
            y = data[target_column(h)]
            base = (q.notna() & y.notna()).to_numpy()
            models = fit_models(data, fm, feats, h, params, quantiles, max_rounds)
            cal = predict_frame(models, data, fm.calib & base, h)
            df = predict_frame(models, data, fm.test & base, h)
            adj = add_intervals(cal, df, h, gamma)
            cal_history[h].append((cal, thr["warning"]))
            rule = choose_rule(cal_history[h], h, step_h)
            thr_hi = float(df["obs"].quantile(0.95))
            common = {"fold": f.name, "test_year": ",".join(map(str, f.test_years)), "horizon_h": h}
            for lev, alpha in LEVELS.items():
                for meth in METHODS:
                    m = interval_metrics(
                        df, df[f"lo{lev}_{meth}"], df[f"hi{lev}_{meth}"], alpha, thr_hi
                    )
                    mets.append({**common, "level": lev, "method": meth, **m})
            for meth in ("quantiles", "cqr", "aci"):
                wis = weighted_interval_score(
                    df["obs"],
                    df["central"],
                    {a: (df[f"lo{lv}_{meth}"], df[f"hi{lv}_{meth}"]) for lv, a in LEVELS.items()},
                )
                mets.append({**common, "level": 0, "method": meth, "wis": wis})
            point = continuous_scores(df["obs"], df["central"], persistence=df["q_now"])
            mets.append({**common, "level": -1, "method": "central", **point})
            for r in RULES:
                ev, _ = alert_counts(df, r, thr["warning"], h, step_h)
                alerts.append(
                    {
                        **common,
                        "rule": r,
                        "chosen": r == rule,
                        "threshold": thr["warning"],
                        **_scores(ev),
                    }
                )
            preds.append(df.assign(fold=f.name, horizon_h=h, hydro_year=years[fm.test & base]))
            log.info(
                "%-7s %4gh  80 %% coverage: quantiles %.2f · CQR %.2f · ACI %.2f  rule %s  "
                "(CQR log adjustment %.3f/%.3f, %.0f s)",
                f.name,
                h,
                coverage(df["obs"], df["lo80_quantiles"], df["hi80_quantiles"]),
                coverage(df["obs"], df["lo80_cqr"], df["hi80_cqr"]),
                coverage(df["obs"], df["lo80_aci"], df["hi80_aci"]),
                rule,
                adj[80][0],
                adj[80][1],
                time.perf_counter() - t0,
            )
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.concat(preds)
    float_cols = pred.select_dtypes("float64").columns
    pred[float_cols] = pred[float_cols].astype("float32")
    pred.to_parquet(out / "predictions.parquet")
    met = pd.DataFrame(mets)
    al = pd.DataFrame(alerts)
    met.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    al.to_csv(out / "warnings.csv", index=False, encoding="utf-8-sig")
    text = summary(pred, met, al, variant, folds, step_h)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")
    return {"predictions": pred, "metrics": met, "warnings": al}


def pooled(pred: pd.DataFrame, al: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics over all the test years pooled together (as in train.py)."""
    rows = []
    for h, part in pred.groupby("horizon_h"):
        thr_hi = float(part["obs"].quantile(0.95))
        for lev, alpha in LEVELS.items():
            for meth in METHODS:
                rows.append(
                    {
                        "horizon_h": h,
                        "level": lev,
                        "method": meth,
                        **interval_metrics(
                            part, part[f"lo{lev}_{meth}"], part[f"hi{lev}_{meth}"], alpha, thr_hi
                        ),
                    }
                )
    ints = pd.DataFrame(rows)
    tot = al.groupby(["horizon_h", "rule"])[["hits", "misses", "false_alarms"]].sum().reset_index()
    chosen = al[al["chosen"]].groupby("horizon_h")[["hits", "misses", "false_alarms"]]
    ch = chosen.sum().reset_index().assign(rule="chosen per fold")
    tot = pd.concat([tot, ch], ignore_index=True)
    for k, num, den in (
        ("pod", "hits", ("hits", "misses")),
        ("far", "false_alarms", ("hits", "false_alarms")),
        ("csi", "hits", ("hits", "misses", "false_alarms")),
    ):
        d = sum(tot[c] for c in den)
        tot[k] = np.where(d > 0, tot[num] / d.where(d > 0, 1), np.nan)
    return ints, tot


def summary(pred, met, al, variant, folds, step_h) -> str:
    ints, tot = pooled(pred, al)
    years = sorted({str(y) for y in pred["hydro_year"].unique()})
    lines = [
        f"=== Variant {variant.name}: uncertainty and warnings "
        f"(years pooled: {', '.join(years)}) ==="
    ]
    for lev in LEVELS:
        sub = ints[ints["level"] == lev]
        for col, title in (
            ("coverage", f"Coverage of the {lev} % interval (target {lev / 100:.2f})"),
            ("coverage_high", f"Coverage of the {lev} % interval at high flows (> p95)"),
            ("median_rel_width", f"Median relative width of the {lev} % interval (× observed)"),
            ("interval_score", f"Interval score of the {lev} % interval (m³/s; lower is better)"),
        ):
            t = sub.pivot(index="method", columns="horizon_h", values=col)
            t.columns = [f"{c:g} h" for c in t.columns]
            lines += ["", f"{title}:", t.round(3).to_string()]
    wis = met[met["level"] == 0].pivot_table(
        index="method", columns="horizon_h", values="wis", aggfunc="mean"
    )
    wis.columns = [f"{c:g} h" for c in wis.columns]
    lines += ["", "WIS, mean of the folds (m³/s; lower is better):", wis.round(3).to_string()]
    for col, title in (("csi", "Warning CSI"), ("pod", "POD"), ("far", "FAR")):
        t = tot.pivot(index="rule", columns="horizon_h", values=col)
        t.columns = [f"{c:g} h" for c in t.columns]
        lines += ["", f"{title} (episodes pooled):", t.round(3).to_string()]
    ch = al[al["chosen"]].pivot(index="fold", columns="horizon_h", values="rule")
    ch.columns = [f"{c:g} h" for c in ch.columns]
    lines += ["", "Rule chosen on the calibration years, per fold:", ch.to_string()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Final evaluation (hold-out)
# --------------------------------------------------------------------------- #
def config_fingerprint(processed: Path, cfg: StationConfig, variant: ModelVariant) -> str:
    """Fingerprint of everything that fixes the system: configuration, features, parameters."""
    d = dataset_dir(processed, cfg, variant)
    parts = [cfg.model_dump_json(), variant.model_dump_json()]
    for f in sorted(
        [d / "feature_analysis" / "feature_selection.csv", *(d / "tuning" / "final").glob("*.json")]
    ):
        if f.exists():
            parts.append(f.read_text(encoding="utf-8"))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def tfg_comparison(pred: pd.DataFrame, start="2025-06-01", end="2026-06-01") -> pd.DataFrame:
    """Metrics over the bachelor thesis test period (June 2025 – May 2026), horizons ≤ 6 h."""
    a, b = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    sub = pred[(pred.index >= a) & (pred.index < b) & (pred["horizon_h"] <= 6)]
    rows = []
    for h, part in sub.groupby("horizon_h"):
        for model in ("central", "lgbm", "ridge", "q_now"):
            s = continuous_scores(part["obs"], part[model], persistence=part["q_now"])
            rows.append(
                {
                    "horizon_h": h,
                    "model": "persistence" if model == "q_now" else model,
                    "rows": len(part),
                    **s,
                }
            )
    return pd.DataFrame(rows)


def final_evaluation(
    cfg: StationConfig,
    variant: ModelVariant,
    processed: Path,
    horizons: list[float],
    force: bool = False,
    **kw,
) -> dict:
    out = dataset_dir(processed, cfg, variant) / "final"
    lock = out / "HOLDOUT_EVALUATED.json"
    fp = config_fingerprint(processed, cfg, variant)
    if lock.exists() and not force:
        prev = json.loads(lock.read_text(encoding="utf-8"))
        raise SystemExit(
            f"The final evaluation was already run on {prev['date']} "
            f"(fingerprint {prev['fingerprint']}).\n"
            "The hold-out must be looked at only once: if it is used again to decide anything, "
            "it is no longer an independent evaluation. (--force allows it anyway, and it is "
            "recorded.)"
        )
    folds = [
        f
        for f in dataset_folds(load_dataset(processed, cfg, variant)[0], cfg, variant)
        if f.name == "final"
    ]
    res = run(cfg, variant, processed, folds, horizons, out, **kw)
    tfg = tfg_comparison(res["predictions"])
    tfg.round(4).to_csv(out / "tfg_comparison.csv", index=False, encoding="utf-8-sig")
    if len(tfg):
        t = tfg.pivot(index="model", columns="horizon_h", values="nse")
        t.columns = [f"{c:g} h" for c in t.columns]
        text = (
            "\nNSE over the thesis test period (June 2025 – May 2026):\n" + t.round(3).to_string()
        )
        print(text)
        with open(out / "summary.txt", "a", encoding="utf-8") as fh:
            fh.write("\n" + text + "\n")
    history = []
    if lock.exists():
        history = json.loads(lock.read_text(encoding="utf-8")).get("history", [])
    entry = {"date": datetime.now(UTC).isoformat(timespec="seconds"), "fingerprint": fp}
    lock.write_text(
        json.dumps({**entry, "history": [*history, entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return res


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--variant", default="long")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--horizons", nargs="+", type=float)
    p.add_argument("--quantiles", nargs="+", type=float, default=list(QUANTILES))
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--rapid", action="store_true")
    return p


def _setup(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    v = cfg.variant(args.variant)
    needed = {round(a / 2, 4) for a in LEVELS.values()} | {
        round(1 - a / 2, 4) for a in LEVELS.values()
    }
    if not needed <= {round(x, 4) for x in args.quantiles}:
        raise SystemExit(f"The quantiles {sorted(needed)} are needed")
    kw = {"quantiles": tuple(args.quantiles), "max_rounds": args.max_rounds, "rapid": args.rapid}
    return cfg, v, kw


def main(argv: list[str] | None = None) -> None:
    p = _parser("llobregat intervals")
    p.add_argument("--folds", nargs="+", type=int, help="Test (hydrological) years")
    args = p.parse_args(argv)
    cfg, v, kw = _setup(args)
    from llobregat.train import choose_folds

    processed = Path(args.processed)
    data, _ = load_dataset(processed, cfg, v)
    folds = choose_folds(dataset_folds(data, cfg, v), args.folds)
    del data
    run(
        cfg,
        v,
        processed,
        folds,
        args.horizons or cfg.horizons_h,
        dataset_dir(processed, cfg, v) / "uncertainty",
        **kw,
    )


def main_final(argv: list[str] | None = None) -> None:
    p = _parser("llobregat final-evaluation")
    p.add_argument("--force", action="store_true", help="Run it again (it is recorded)")
    args = p.parse_args(argv)
    cfg, v, kw = _setup(args)
    final_evaluation(
        cfg, v, Path(args.processed), args.horizons or cfg.horizons_h, force=args.force, **kw
    )

"""Feature analysis before training.

Everything is computed on the development period only (the training period of the first
fold), so no decision depends on the calibration, test or hold-out years.

  1. Spearman correlation of each feature with the target of each horizon: with the flow
     change (y_h − Q_t), which is what the model has to predict, and with the level.
  2. Redundancy: Spearman matrix between features and groups of near-identical features
     (|ρ| ≥ threshold, complete linkage: every pair in the group exceeds it). One feature of
     each group is kept; constant features and those with too many gaps are dropped too.
     For trees, duplicates do not hurt the forecast, but they split the importance and slow
     training down; for Ridge, they make it unstable.
  3. Travel time to the target station, peak to peak: for each flood (peak above the 99th
     percentile, at least 72 h apart), where the maximum of each upstream gauge lies within
     the previous 48 h. It is compared with the lag of maximum correlation of the rises.

  uv run llobregat analyse-features [--variant long|full|all]

Output in data/processed/<station>/<variant>/feature_analysis/: feature_selection.csv,
target_correlation.csv, duplicates.csv, travel_times.csv, travel_time_peaks.csv,
correlations.parquet and summary.txt.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import ModelVariant, StationConfig, load_station_config
from llobregat.dataset import feature_block
from llobregat.experiment import (
    OWN_FLOW,
    dataset_dir,
    dataset_folds,
    development_mask,
    feature_columns,
    load_dataset,
    target_column,
)
from llobregat.features import estimate_travel_time
from llobregat.metrics import find_runs

log = logging.getLogger("analyse-features")

PROTECTED = (OWN_FLOW,)  # never dropped: persistence and the target transforms need it
MAX_NAN_FRACTION = 0.5


# --------------------------------------------------------------------------- #
# Feature groups (one source = a gauge, a rain gauge, a reservoir...)
# --------------------------------------------------------------------------- #
_GROUP_PATTERNS = [
    (re.compile(r"^q_"), lambda m: "own"),
    (re.compile(r"^up_(.+)_(?:lag_[^_]+|diff_1h)$"), lambda m: f"flow:{m.group(1)}"),
    (
        re.compile(r"^res_(.+)_(out|vol)_(?:now|mean_6h|diff_3h|diff_6h|diff_24h)$"),
        lambda m: f"{'outflow' if m.group(2) == 'out' else 'volume'}:{m.group(1)}",
    ),
    (re.compile(r"^rain_st_(.+)_sum_[^_]+$"), lambda m: f"rain:{m.group(1)}"),
    (re.compile(r"^rain_sb_(.+)_(?:sum_[^_]+|imax_6h)$"), lambda m: f"subbasin:{m.group(1)}"),
    (re.compile(r"^(?:rain_basin_|api_|hours_since_rain)"), lambda m: "basin_rain"),
    (re.compile(r"^doy_"), lambda m: "calendar"),
]
ALWAYS_KEEP_GROUPS = ("own", "calendar")


def feature_group(col: str) -> str:
    """Source of a feature: 'flow:castellbell', 'rain:EA049', 'own'..."""
    for pat, fn in _GROUP_PATTERNS:
        m = pat.match(col)
        if m:
            return fn(m)
    return "other"


def feature_groups(cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(feature_group(c), []).append(c)
    return groups


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #
def sample_rows(n: int, size: int | None, seed: int = 0) -> np.ndarray:
    if size is None or n <= size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def spearman_matrix(X: pd.DataFrame, min_periods: int = 100) -> pd.DataFrame:
    """Pairwise Spearman (gaps in one feature do not remove rows from the others)."""
    return X.rank().corr(method="pearson", min_periods=min_periods)


def target_correlations(X: pd.DataFrame, data: pd.DataFrame, horizons: list[float]) -> pd.DataFrame:
    ranks = X.rank()
    out = pd.DataFrame(index=X.columns)
    q = data[OWN_FLOW]
    for h in horizons:
        y = data[target_column(h)]
        out[f"rho_change_{h:g}h"] = ranks.corrwith((y - q).rank())
        out[f"rho_level_{h:g}h"] = ranks.corrwith(y.rank())
    delta_cols = [c for c in out.columns if c.startswith("rho_change")]
    out["rho_change_abs_mean"] = out[delta_cols].abs().mean(axis=1)
    return out


def near_duplicate_groups(corr: pd.DataFrame, threshold: float) -> np.ndarray:
    """Group label per feature (complete linkage on 1 − |ρ|)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = 1 - corr.abs().to_numpy(dtype=float)
    d = np.where(np.isfinite(d), d, 1.0)
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2, 0, 1)
    if len(d) < 2:
        return np.ones(len(d), dtype=int)
    z = linkage(squareform(d, checks=False), method="complete")
    return fcluster(z, t=1 - threshold, criterion="distance")


def choose_features(
    corr: pd.DataFrame,
    nan_fraction: pd.Series,
    std: pd.Series,
    threshold: float,
    protected: tuple[str, ...] = PROTECTED,
) -> pd.DataFrame:
    """Decide which features are kept. From each group of duplicates: the protected one,
    if any; otherwise the one with fewest gaps; on a tie, the first one (the simplest)."""
    cols = list(corr.columns)
    out = pd.DataFrame(
        {
            "feature": cols,
            "block": [feature_block(c) for c in cols],
            "group": [feature_group(c) for c in cols],
            "nan_pct": (nan_fraction.reindex(cols) * 100).round(2).to_numpy(),
        }
    )
    out["keep"] = 1
    out["reason"] = ""
    const = (std.reindex(cols).fillna(0) == 0).to_numpy()
    sparse = (nan_fraction.reindex(cols) > MAX_NAN_FRACTION).to_numpy()
    prot = out["feature"].isin(protected).to_numpy()
    out.loc[const & ~prot, ["keep", "reason"]] = [0, "constant"]
    out.loc[sparse & ~prot & ~const, ["keep", "reason"]] = [0, "too many gaps"]

    alive = out["keep"].to_numpy() == 1
    labels = np.zeros(len(cols), dtype=int)
    sub = corr.loc[alive, alive]
    labels[alive] = near_duplicate_groups(sub, threshold)
    out["duplicate_group"] = labels
    order = {c: i for i, c in enumerate(cols)}
    for _, part in out[alive].groupby("duplicate_group"):
        if len(part) < 2:
            continue
        cand = part.assign(
            _prot=part["feature"].isin(protected),
            _order=part["feature"].map(order),
        ).sort_values(["_prot", "nan_pct", "_order"], ascending=[False, True, True])
        keep = cand["feature"].iloc[0]
        for v in cand["feature"].iloc[1:]:
            i = order[v]
            out.loc[i, ["keep", "reason"]] = [0, f"duplicate of {keep}"]
    out.loc[~alive, "duplicate_group"] = 0
    return out


# --------------------------------------------------------------------------- #
# Peak-to-peak travel time
# --------------------------------------------------------------------------- #
def flood_peaks(q: pd.Series, quantile: float = 0.99, min_sep_steps: int = 144) -> list[int]:
    """Position of the peak of each flood: runs above the quantile, merged when they are
    less than min_sep_steps apart."""
    v = q.to_numpy(dtype=float)
    thr = np.nanquantile(v, quantile)
    runs = find_runs(np.nan_to_num(v, nan=-np.inf) > thr)
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < min_sep_steps:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [s + int(np.nanargmax(v[s : e + 1])) for s, e in merged]


def peak_travel_times(
    q: pd.Series,
    up: pd.Series,
    peaks: list[int],
    max_lag_steps: int,
    step_h: float,
    min_quantile: float = 0.9,
) -> pd.DataFrame:
    """Lag between the upstream peak (within the previous max_lag_steps) and the downstream one."""
    u = up.to_numpy(dtype=float)
    thr = np.nanquantile(u, min_quantile) if np.isfinite(u).any() else np.inf
    rows = []
    for p in peaks:
        lo = max(0, p - max_lag_steps)
        seg = u[lo : p + 1]
        if not np.isfinite(seg).any():
            continue
        j = int(np.nanargmax(seg))
        valid = j > 0 and seg[j] >= thr  # maximum at the edge: the upstream peak is earlier
        rows.append(
            {
                "peak": q.index[p],
                "peak_flow": float(q.iloc[p]),
                "upstream_peak": up.index[lo + j],
                "upstream_peak_flow": float(seg[j]),
                "lag_h": (len(seg) - 1 - j) * step_h,
                "valid": bool(valid),
            }
        )
    return pd.DataFrame(rows)


def travel_time_table(
    data: pd.DataFrame, dev: np.ndarray, step_h: float, max_lag_h: float = 48
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = data.loc[dev, OWN_FLOW]
    peaks = flood_peaks(q, 0.99, min_sep_steps=round(72 / step_h))
    max_steps = round(max_lag_h / step_h)
    per_event, summary = [], []
    stations = sorted({m.group(1) for c in data.columns if (m := re.match(r"^up_(.+)_lag_0h$", c))})
    for st in stations:
        up = data.loc[dev, f"up_{st}_lag_0h"]
        ev = peak_travel_times(q, up, peaks, max_steps, step_h)
        if ev.empty:
            continue
        per_event.append(ev.assign(station=st))
        ok = ev[ev["valid"]]["lag_h"]
        lag_x, corr_x = estimate_travel_time(up, q, round(24 / step_h), min_quantile=0.95)
        summary.append(
            {
                "station": st,
                "floods": len(ev),
                "valid": len(ok),
                "lag_median_h": ok.median() if len(ok) else np.nan,
                "lag_p25_h": ok.quantile(0.25) if len(ok) else np.nan,
                "lag_p75_h": ok.quantile(0.75) if len(ok) else np.nan,
                "lag_correlation_h": lag_x * step_h,
                "corr_max": round(float(corr_x.max()), 3) if corr_x.notna().any() else np.nan,
            }
        )
    ev_all = pd.concat(per_event, ignore_index=True) if per_event else pd.DataFrame()
    return pd.DataFrame(summary), ev_all


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(
    cfg: StationConfig,
    variant: ModelVariant,
    processed: Path,
    sample: int | None = 150_000,
    threshold: float = 0.995,
) -> dict:
    data, _ = load_dataset(processed, cfg, variant)
    folds = dataset_folds(data, cfg, variant)
    dev = development_mask(folds)
    f0 = folds[0].fold
    feats = feature_columns(data)
    has_y = data[target_column(cfg.horizons_h[0])].notna().to_numpy() & data[OWN_FLOW].notna()
    rows = np.flatnonzero(dev & has_y.to_numpy())
    rows = rows[sample_rows(len(rows), sample)]
    X = data.iloc[rows][feats]
    log.info(
        "%s: development period %s–%s (%d rows; sample of %d)",
        variant.name,
        f0.train_years[0],
        f0.train_years[-1],
        int(dev.sum()),
        len(X),
    )
    corr = spearman_matrix(X)
    tcorr = target_correlations(X, data.iloc[rows], cfg.horizons_h)
    nan_frac = data.loc[dev, feats].isna().mean()
    std = X.std()
    sel = choose_features(corr, nan_frac, std, threshold)
    sel = sel.merge(tcorr[["rho_change_abs_mean"]].round(4), left_on="feature", right_index=True)
    step_h = cfg.resolution_min / 60
    tt, tt_ev = travel_time_table(data, dev, step_h)

    out = dataset_dir(processed, cfg, variant) / "feature_analysis"
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "feature_selection.csv", index=False, encoding="utf-8-sig")
    tcorr.round(4).rename_axis("feature").to_csv(
        out / "target_correlation.csv", encoding="utf-8-sig"
    )
    dup = sel[sel["duplicate_group"].map(sel["duplicate_group"].value_counts()) > 1]
    dup = dup[dup["duplicate_group"] > 0].sort_values(["duplicate_group", "keep"])
    dup.to_csv(out / "duplicates.csv", index=False, encoding="utf-8-sig")
    corr.astype("float32").to_parquet(out / "correlations.parquet")
    tt.round(2).to_csv(out / "travel_times.csv", index=False, encoding="utf-8-sig")
    tt_ev.to_csv(out / "travel_time_peaks.csv", index=False, encoding="utf-8-sig")

    text = summary_text(variant, f0, sel, tcorr, tt, cfg.horizons_h, threshold)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")
    return {"selection": sel, "target_correlation": tcorr, "travel_times": tt}


def summary_text(variant, f0, sel, tcorr, tt, horizons, threshold) -> str:
    kept = sel[sel["keep"] == 1]
    lines = [
        f"=== Variant {variant.name}: features (development {f0.train_years[0]}–"
        f"{f0.train_years[-1]}) ===",
        f"Features: {len(sel)} → {len(kept)} kept "
        f"(constant {int((sel['reason'] == 'constant').sum())}, "
        f"too many gaps {int((sel['reason'] == 'too many gaps').sum())}, "
        f"duplicates with |ρ| ≥ {threshold} "
        f"{int(sel['reason'].str.startswith('duplicate').sum())})",
        "",
        "By block (kept / total):",
    ]
    per_block = sel.groupby("block")["keep"].agg(["sum", "size"])
    for b, r in per_block.iterrows():
        lines.append(f"  {b:<22} {int(r['sum']):>4} / {int(r['size'])}")
    for h in (horizons[0], horizons[len(horizons) // 2], horizons[-1]):
        col = f"rho_change_{h:g}h"
        top = tcorr[col].abs().sort_values(ascending=False).head(10)
        lines += ["", f"Most correlated with the flow change at {h:g} h (Spearman):"]
        for v in top.index:
            lines.append(f"  {v:<40} {tcorr.loc[v, col]:+.3f}")
    if len(tt):
        lines += ["", "Travel time to the target station (hours):"]
        lines.append(
            f"  {'station':<22}{'peaks':>6}{'valid':>8}{'median':>9}{'p25–p75':>12}{'correl.':>9}"
        )
        for r in tt.sort_values("lag_median_h").itertuples():
            iqr = f"{r.lag_p25_h:.1f}–{r.lag_p75_h:.1f}" if np.isfinite(r.lag_p25_h) else "—"
            lines.append(
                f"  {r.station:<22}{r.floods:>6}{r.valid:>8}"
                f"{r.lag_median_h:>9.1f}{iqr:>12}{r.lag_correlation_h:>9.1f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat analyse-features",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--variant", default="all")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--sample", type=int, default=150_000, help="Rows for the correlations")
    p.add_argument("--duplicate-threshold", type=float, default=0.995)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    variants = cfg.variants if args.variant == "all" else [cfg.variant(args.variant)]
    pd.set_option("display.width", 200)
    for v in variants:
        run(cfg, v, Path(args.processed), args.sample, args.duplicate_threshold)

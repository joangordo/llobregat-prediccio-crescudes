"""Final report: hold-out metrics, warning lead times, episodes and figures.

Reads the outputs that already exist (nothing is retrained and no decision is taken here):
    uncertainty/predictions.parquet  cross-validation 2020–2024 of the final system
    final/predictions.parquet        hold-out 2025–2026 (evaluated once)
    final/warnings.csv               warning threshold of the hold-out fold
    feature_analysis/travel_times.csv  travel times
    operational/source_importance.csv  (or models/<lgbm>/source_importance.csv)

    uv run llobregat report --variant long

Writes data/processed/<station>/<variant>/report/: point_metrics.csv, lead_times.csv,
episodes.csv, coverage_by_flow.csv, figures/*.png and report.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from llobregat.config import ModelVariant, StationConfig, load_station_config
from llobregat.experiment import dataset_dir
from llobregat.metrics import continuous_scores, coverage, episode_table, find_runs

log = logging.getLogger("report")

MODELS = {
    "central": "Ensemble (final)",
    "lgbm": "LightGBM",
    "ridge": "Ridge",
    "q_now": "Persistence",
}
TIERS = {"alert": "central", "pre_alert": "hi90_cqr"}
# Reference categorical palette (fixed order) plus a neutral for the baseline
COLORS = {
    "central": "#2a78d6",
    "lgbm": "#eb6834",
    "ridge": "#1baf7a",
    "q_now": "#8a8985",
    "obs": "#0b0b0b",
    "band": "#2a78d6",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def observed_series(pred: pd.DataFrame) -> pd.Series:
    """Observed flow on a regular grid, from q_now at every issue time."""
    s = pred["q_now"].groupby(level=0).first().sort_index()
    step = s.index.to_series().diff().median()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=step))


def target_axis(part: pd.DataFrame, col: str, h: float, grid: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(part[col].to_numpy(dtype=float), index=part.index + pd.Timedelta(hours=h))
    return s[~s.index.duplicated()].reindex(grid)


def threshold_from(final_dir: Path) -> float:
    al = pd.read_csv(final_dir / "warnings.csv")
    return float(al["threshold"].iloc[0])


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def point_metrics(pred: pd.DataFrame, threshold: float, step_h: float) -> pd.DataFrame:
    """Continuous and peak metrics per horizon and model, on the target-time axis."""
    rows = []
    obs = observed_series(pred)
    for h, part in pred.groupby("horizon_h"):
        grid = obs.index
        o = obs
        hi = part["obs"] > part["obs"].quantile(0.95)
        window = round(max(12.0, h + 6.0) / step_h)
        for col, name in MODELS.items():
            s = continuous_scores(part["obs"], part[col], persistence=part["q_now"])
            err = part.loc[hi, col] - part.loc[hi, "obs"]
            p = target_axis(part, col, h, grid)
            ep = episode_table(o, p, threshold, search_steps=window)
            rows.append(
                {
                    "horizon_h": h,
                    "model": name,
                    "nse": s["nse"],
                    "kge": s["kge"],
                    "rmse_m3s": s["rmse"],
                    "persistence_index": s.get("pi", np.nan),
                    "high_flow_bias_pct": 100 * err.mean() / part.loc[hi, "obs"].mean(),
                    "episodes": len(ep),
                    "peak_bias_pct": 100 * ep["peak_rel_error"].median() if len(ep) else np.nan,
                    "peak_abs_error_pct": 100 * ep["peak_rel_error"].abs().median()
                    if len(ep)
                    else np.nan,
                    "peak_delay_h": ep["peak_timing_error_h"].median() if len(ep) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def lead_times(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Earliest warning per observed episode, combining every horizon.

    An episode is a run of observed flow above the threshold. A forecast issued at t for
    t + h warns of the episode if it is above the threshold and its target time falls
    between the start of the episode and its peak, and if at t the flow was still below the
    threshold (a warning issued once the river is already over it is not a warning). The
    lead time is the start of the episode minus the earliest such issue time.
    """
    obs = observed_series(pred)
    o = obs.to_numpy(dtype=float)
    rows = []
    for s, e in find_runs(np.nan_to_num(o, nan=-np.inf) > threshold):
        peak = s + int(np.nanargmax(o[s : e + 1]))
        start, peak_t = obs.index[s], obs.index[peak]
        row = {
            "start": start,
            "peak_time": peak_t,
            "peak_m3s": o[peak],
            "duration_h": (obs.index[e] - start).total_seconds() / 3600,
        }
        for tier, col in TIERS.items():
            best_t, best_h = None, np.nan
            for h, part in pred.groupby("horizon_h"):
                tgt = part.index + pd.Timedelta(hours=h)
                ok = (
                    (tgt >= start)
                    & (tgt <= peak_t)
                    & (part[col].to_numpy() > threshold)
                    & (part["q_now"].to_numpy() <= threshold)
                )
                if ok.any():
                    t = part.index[ok].min()
                    if best_t is None or t < best_t:
                        best_t, best_h = t, h
            row[f"{tier}_issue"] = best_t
            row[f"{tier}_horizon_h"] = best_h
            row[f"{tier}_lead_h"] = (
                (start - best_t).total_seconds() / 3600 if best_t is not None else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def episodes(
    pred: pd.DataFrame, threshold: float, step_h: float, top: int, label: str
) -> pd.DataFrame:
    """Largest episodes: observed peak, forecast peak, delay and interval cover per horizon."""
    obs = observed_series(pred)
    ep0 = episode_table(obs, obs, threshold)
    if ep0.empty:
        return ep0
    ep0 = ep0.nlargest(top, "peak_obs")
    rows = []
    for h, part in pred.groupby("horizon_h"):
        window = round(max(12.0, h + 6.0) / step_h)
        cols = {
            c: target_axis(part, c, h, obs.index)
            for c in ("central", "lo80_cqr", "hi80_cqr", "lo90_cqr", "hi90_cqr")
            if c in part
        }
        ep = episode_table(obs, cols["central"], threshold, search_steps=window)
        ep = ep[ep["peak_time"].isin(ep0["peak_time"])]
        for r in ep.itertuples(index=False):
            out = {
                "set": label,
                "start": r.start,
                "peak_time": r.peak_time,
                "peak_obs_m3s": r.peak_obs,
                "horizon_h": h,
                "peak_pred_m3s": r.peak_pred,
                "peak_error_pct": 100 * r.peak_rel_error,
                "peak_delay_h": r.peak_timing_error_h,
            }
            for lev in (80, 90):
                lo, hi = cols.get(f"lo{lev}_cqr"), cols.get(f"hi{lev}_cqr")
                if lo is not None:
                    out[f"peak_in_{lev}"] = bool(lo[r.peak_time] <= r.peak_obs <= hi[r.peak_time])
                    out[f"hi{lev}_at_peak_m3s"] = float(hi[r.peak_time])
            rows.append(out)
    return pd.DataFrame(rows)


def coverage_by_flow(pred: pd.DataFrame) -> pd.DataFrame:
    edges = pred["obs"].quantile([0.5, 0.9, 0.99]).to_numpy()
    labels = ["< p50", "p50–p90", "p90–p99", "> p99"]
    cls = pd.cut(pred["obs"], [-np.inf, *edges, np.inf], labels=labels)
    rows = []
    for (h, c), part in pred.groupby(["horizon_h", cls], observed=True):
        rows.append(
            {
                "horizon_h": h,
                "flow_class": c,
                "rows": len(part),
                "coverage_80": coverage(part["obs"], part["lo80_cqr"], part["hi80_cqr"]),
                "coverage_90": coverage(part["obs"], part["lo90_cqr"], part["hi90_cqr"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK2,
            "text.color": INK,
            "figure.dpi": 100,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    return plt


def year_span(pred: pd.DataFrame) -> str:
    y = sorted(int(v) for v in pred["hydro_year"].unique())
    return f"{y[0]}–{y[-1]}" if len(y) > 1 else str(y[0])


def fig_skill(cv: pd.DataFrame, ho: pd.DataFrame, path: Path, spans=("", "")) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, df, title in (
        (axes[0], cv, f"Cross-validation {spans[0]}"),
        (axes[1], ho, f"Hold-out {spans[1]}"),
    ):
        for col, name in MODELS.items():
            d = df[df["model"] == name].sort_values("horizon_h")
            ax.plot(d["horizon_h"], d["nse"], marker="o", ms=5, lw=2, color=COLORS[col], label=name)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Lead time (h)")
        ax.set_xticks(sorted(df["horizon_h"].unique()))
        _style(ax)
    axes[0].set_ylabel("NSE")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_hydrograph(
    pred: pd.DataFrame,
    peak_time: pd.Timestamp,
    threshold: float,
    path: Path,
    title: str,
    horizons=(1, 6, 12, 24),
) -> None:
    plt = _plt()
    import matplotlib.dates as mdates

    obs = observed_series(pred)
    a, b = peak_time - pd.Timedelta(days=2), peak_time + pd.Timedelta(days=2)
    hs = [h for h in horizons if h in set(pred["horizon_h"])]
    fig, axes = plt.subplots(1, len(hs), figsize=(3.2 * len(hs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs, strict=True):
        part = pred[pred["horizon_h"] == h]
        grid = obs.index
        sel = (grid >= a) & (grid <= b)
        t = grid[sel]
        get = {
            c: target_axis(part, c, h, grid)[sel]
            for c in ("central", "lo80_cqr", "hi80_cqr", "lo90_cqr", "hi90_cqr")
        }
        ax.fill_between(
            t,
            get["lo90_cqr"],
            get["hi90_cqr"],
            color=COLORS["band"],
            alpha=0.12,
            lw=0,
            label="90 % interval",
        )
        ax.fill_between(
            t,
            get["lo80_cqr"],
            get["hi80_cqr"],
            color=COLORS["band"],
            alpha=0.22,
            lw=0,
            label="80 % interval",
        )
        ax.plot(t, obs[sel], color=COLORS["obs"], lw=1.6, label="Observed")
        ax.plot(t, get["central"], color=COLORS["central"], lw=2, label="Forecast")
        ax.axhline(threshold, color=INK2, lw=1, ls="--")
        ax.set_title(f"+{h:g} h", loc="left")
        # A label every other day ("13 Sep") and a minor tick every day: readable in 4 panels
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz="UTC"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz="UTC"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(tz="UTC"))
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        _style(ax)
    axes[0].set_ylabel("Flow (m³/s)")
    axes[0].text(t[0], threshold, " warning threshold", va="bottom", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.suptitle(title, x=0.01, y=1.01, ha="left", fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def fig_coverage(cov: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    hs = sorted(cov["horizon_h"].unique())
    classes = (
        list(cov["flow_class"].cat.categories)
        if hasattr(cov["flow_class"], "cat")
        else list(dict.fromkeys(cov["flow_class"]))
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    ramp = ["#c6dbf5", "#8ab6ea", "#4a90de", "#1f5fae"]  # one hue, light → dark
    width = 0.8 / len(classes)
    for ax, lev in zip(axes, (80, 90), strict=True):
        for i, c in enumerate(classes):
            d = cov[cov["flow_class"] == c].set_index("horizon_h").reindex(hs)
            x = np.arange(len(hs)) + (i - (len(classes) - 1) / 2) * width
            ax.bar(x, d[f"coverage_{lev}"], width * 0.9, color=ramp[i % len(ramp)], label=c)
        ax.axhline(lev / 100, color=INK, lw=1.2, ls="--")
        ax.set_xticks(np.arange(len(hs)), [f"+{h:g} h" for h in hs])
        ax.set_title(f"{lev} % interval (nominal {lev / 100:.2f})", loc="left")
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].set_ylabel("Observed coverage")
    axes[1].legend(
        title="Observed flow", frameon=False, fontsize=8, title_fontsize=8, loc="lower left"
    )
    fig.savefig(path)
    plt.close(fig)


def fig_importance(imp: pd.DataFrame, path: Path, top: int = 15) -> None:
    plt = _plt()
    hcol = "horizon_h"
    table = imp.groupby(["group", hcol])["mse_increase"].median().unstack(hcol) * 100
    table = table.loc[table.max(axis=1).sort_values(ascending=False).index[:top]]
    fig, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 3, 0.36 * len(table) + 1.2))
    vals = np.clip(table.to_numpy(), 0, None)
    im = ax.imshow(np.log1p(vals), cmap="Blues", aspect="auto")
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        dark = np.log1p(max(v, 0)) > 0.6 * np.log1p(np.nanmax(vals))
        ax.text(
            j,
            i,
            f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if dark else INK,
        )
    ax.set_xticks(range(table.shape[1]), [f"+{h:g} h" for h in table.columns])
    ax.set_yticks(range(len(table)), table.index)
    ax.set_title("Increase in MSE when a source is shuffled (%)", loc="left")
    ax.tick_params(length=0, labelsize=9, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    del im
    fig.savefig(path)
    plt.close(fig)


def fig_travel(tt: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    d = tt.dropna(subset=["lag_median_h"]).sort_values("lag_median_h")
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(d) + 1))
    y = np.arange(len(d))
    ax.barh(y, d["lag_median_h"], 0.6, color=COLORS["central"])
    if {"lag_p25_h", "lag_p75_h"} <= set(d):
        lo, hi = d["lag_p25_h"], d["lag_p75_h"]
    else:
        lo = hi = d["lag_median_h"]
    ax.errorbar(
        d["lag_median_h"],
        y,
        xerr=[d["lag_median_h"] - lo, hi - d["lag_median_h"]],
        fmt="none",
        ecolor=INK2,
        capsize=3,
        lw=1,
    )
    for yi, v, top in zip(y, d["lag_median_h"], hi.fillna(d["lag_median_h"]), strict=True):
        ax.text(max(v, top) + 0.4, yi, f"{v:.1f} h", fontsize=8, color=INK, va="center")
    ax.set_yticks(y, d["station"])
    ax.set_xlabel("Peak-to-peak travel time to the outlet (h, median and IQR)")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def fig_lead(lt: pd.DataFrame, path: Path) -> None:
    """Share of episodes warned at least x hours before the observed threshold crossing."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    n = len(lt)
    leads = {tier: lt[f"{tier}_lead_h"].dropna().to_numpy() for tier in TIERS}
    hi = max([v.max() for v in leads.values() if len(v)] + [1.0])
    lo = min([v.min() for v in leads.values() if len(v)] + [0.0])
    x = np.arange(np.floor(lo) - 0.5, hi + 1.0, 0.5)
    labels = {"alert": "Alert (point forecast)", "pre_alert": "Pre-alert (90 % upper bound)"}
    colors = {"alert": COLORS["central"], "pre_alert": COLORS["lgbm"]}
    for tier, v in leads.items():
        share = [(v >= xi).sum() / n for xi in x]
        ax.step(
            x,
            share,
            where="post",
            lw=2,
            color=colors[tier],
            label=f"{labels[tier]}: {len(v)}/{n} episodes before the peak",
        )
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Warning issued at least x hours before the threshold crossing")
    ax.set_ylabel("Share of episodes")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _importance_table(base: Path) -> pd.DataFrame | None:
    op = base / "operational" / "source_importance.csv"
    if op.exists():
        return pd.read_csv(op)
    for d in sorted((base / "models").glob("lgbm*")):
        f = d / "source_importance.csv"
        if f.exists():
            return pd.read_csv(f)
    return None


def run(cfg: StationConfig, variant: ModelVariant, processed: Path, top: int = 10) -> Path:
    base = dataset_dir(processed, cfg, variant)
    out = base / "report"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    step_h = cfg.resolution_min / 60
    ho = pd.read_parquet(base / "final" / "predictions.parquet")
    cv = pd.read_parquet(base / "uncertainty" / "predictions.parquet")
    thr = threshold_from(base / "final")
    # Thresholds of the CV folds are not stored with the predictions: use the hold-out one
    # (train q99 up to 2024), only for the event-based columns of the CV table.
    pm_ho = point_metrics(ho, thr, step_h).assign(set="hold-out")
    pm_cv = point_metrics(cv, thr, step_h).assign(set="cross-validation")
    pm = pd.concat([pm_cv, pm_ho], ignore_index=True)
    pm.round(4).to_csv(out / "point_metrics.csv", index=False)
    lt = lead_times(ho, thr)
    lt.to_csv(out / "lead_times.csv", index=False)
    ep = pd.concat(
        [
            episodes(ho, thr, step_h, top, "hold-out"),
            episodes(cv, thr, step_h, 3, "cross-validation"),
        ],
        ignore_index=True,
    )
    ep.to_csv(out / "episodes.csv", index=False, float_format="%.3f")
    cov = coverage_by_flow(ho)
    cov.round(4).to_csv(out / "coverage_by_flow.csv", index=False)

    spans = (year_span(cv), year_span(ho))
    fig_skill(pm_cv, pm_ho, figs / "skill_by_horizon.png", spans)
    fig_coverage(cov, figs / "coverage_by_flow.png")
    if len(lt):
        fig_lead(lt, figs / "lead_times.png")
    hydro = []
    for label, pred in (("hold-out", ho), ("cross-validation", cv)):
        sub = (
            ep[ep["set"] == label]
            .drop_duplicates("peak_time")
            .nlargest(3 if label == "hold-out" else 1, "peak_obs_m3s")
        )
        for r in sub.itertuples(index=False):
            name = f"hydrograph_{r.peak_time:%Y%m%d}.png"
            fig_hydrograph(
                pred,
                r.peak_time,
                thr,
                figs / name,
                f"{r.peak_time:%d %b %Y}: observed peak {r.peak_obs_m3s:.0f} m³/s ({label})",
            )
            hydro.append(name)
    imp = _importance_table(base)
    if imp is not None:
        fig_importance(imp, figs / "source_importance.png")
    tt_f = base / "feature_analysis" / "travel_times.csv"
    if tt_f.exists():
        tt = pd.read_csv(tt_f)
        fig_travel(tt, figs / "travel_times.png")

    (out / "report.md").write_text(
        markdown(pm, lt, ep, cov, thr, hydro, imp is not None, tt_f.exists(), spans),
        encoding="utf-8",
    )
    print(f"Report saved to {out}")
    return out


def markdown(pm, lt, ep, cov, thr, hydro, has_imp, has_tt, spans=("", "")) -> str:
    def pivot(df, value, fmt="{:.3f}"):
        t = df.pivot(index="model", columns="horizon_h", values=value)
        t = t.reindex([m for m in MODELS.values() if m in t.index])
        t.columns = [f"+{h:g} h" for h in t.columns]
        return md_table(t.reset_index(), fmt)

    ho = pm[pm["set"] == "hold-out"]
    cv = pm[pm["set"] == "cross-validation"]
    parts = [
        "# Results report",
        "",
        f"Warning threshold (training q99): **{thr:.0f} m³/s**.",
        "",
        "## Skill by lead time",
        "",
        "![Skill](figures/skill_by_horizon.png)",
        "",
        f"### NSE, cross-validation {spans[0]} (pooled over the test years)",
        "",
        pivot(cv, "nse"),
        "",
        f"### NSE, hold-out {spans[1]}",
        "",
        pivot(ho, "nse"),
        "",
        "### Median peak error on the hold-out (%)",
        "",
        pivot(ho, "peak_bias_pct", "{:+.1f}"),
        "",
        "### Median peak delay on the hold-out (h)",
        "",
        pivot(ho, "peak_delay_h", "{:+.1f}"),
        "",
    ]
    if len(lt):
        n = len(lt)

        def describe(tier):
            v = lt[f"{tier}_lead_h"].dropna()
            if v.empty:
                return f"0/{n} episodes"
            return (
                f"{len(v)}/{n} episodes warned before the peak; median lead "
                f"{v.median():+.1f} h before the threshold crossing "
                f"({(v > 0).sum()} before the crossing)"
            )

        parts += [
            "## Warning lead time (hold-out)",
            "",
            f"- Alert (point forecast above the threshold): {describe('alert')}.",
            f"- Pre-alert (upper bound of the 90 % interval above it): {describe('pre_alert')}.",
            "",
            "![Lead times](figures/lead_times.png)",
            "",
        ]
    if len(ep):
        e = ep[ep["set"] == "hold-out"]
        if len(e):
            w = e.pivot_table(
                index=["peak_time", "peak_obs_m3s"], columns="horizon_h", values="peak_error_pct"
            ).sort_index(level=1, ascending=False)
            w.columns = [f"+{h:g} h" for h in w.columns]
            w = w.reset_index()
            w["peak_time"] = w["peak_time"].dt.strftime("%Y-%m-%d %H:%M")
            w["peak_obs_m3s"] = w["peak_obs_m3s"].map("{:.0f}".format)
            parts += [
                "## Largest hold-out episodes: peak error of the forecast (%)",
                "",
                md_table(w, "{:+.0f}"),
                "",
            ]
    parts += [
        "## Interval coverage by flow class (hold-out, CQR)",
        "",
        "![Coverage](figures/coverage_by_flow.png)",
        "",
    ]
    parts += [f"![Episode](figures/{h})\n" for h in hydro]
    if has_imp:
        parts += ["## Source importance", "", "![Importance](figures/source_importance.png)", ""]
    if has_tt:
        parts += ["## Travel times", "", "![Travel times](figures/travel_times.png)", ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--variant", default="long")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--top", type=int, default=10, help="Hold-out episodes in the table")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    run(cfg, cfg.variant(args.variant), Path(args.processed), args.top)

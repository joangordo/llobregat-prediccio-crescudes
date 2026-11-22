"""30-minute data table of each variant (features + targets).

Sources (all in UTC, labelled with the END of the 30-min interval):
  target       flow at Sant Joan Despí, 30-min mean of the clean readings (qc-apply).
               Gaps are never filled.
  flows        upstream gauges of the variant: 30-min mean of the clean readings; gaps of up
               to 3 h (short_gap_intervals) are filled by linear interpolation.
  reservoirs   outflow and volume of the PREVIOUS complete DAY (official daily series). In
               real time they are computed the same way (mean of the previous day from the
               5-min data), so the model is trained with what it will have in operation.
  rain         XEMA (mm per 30 min; hourly readings before 2014 split into two half hours;
               splices ZE<-Z8 and YO<-WY) and ACA rain gauges (mm every 5 min, already shifted
               to the end of the interval) summed to 30 min. Not interpolated.
The inputs of each variant are those chosen by llobregat splits (coverage >= 85 %).

Features: llobregat.features.build_features (own flow, upstream flows with lags, reservoirs,
rain per gauge, per sub-basin and for the basin, antecedent precipitation index and
calendar). Targets: y_<h>h = mean flow of the half hour ending at t + h.

  uv run llobregat build-dataset [--variant long|full|all]

Output in data/processed/<station>/<variant>/: dataset.parquet, meta.json, nans.csv,
travel_times_xcorr.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from llobregat.config import ModelVariant, StationConfig, load_station_config
from llobregat.features import FeatureSpec, build_features, estimate_travel_time
from llobregat.ingest.aca_opendata import safe_name
from llobregat.ingest.xema_history import load_xema_history
from llobregat.qc import fill_short_gaps
from llobregat.resample import splice, to_grid, xema_period_start_to_end

log = logging.getLogger("dataset")

FREQ = "30min"
WARMUP = pd.Timedelta(days=30)  # data before the start for the lags and the antecedent rain
MAX_RAIN_5MIN_MM = 30.0  # more than 30 mm in 5 min (360 mm/h) is not plausible
MAX_RAIN_30MIN_MM = 100.0


# --------------------------------------------------------------------------- #
# Sources on the 30-min grid
# --------------------------------------------------------------------------- #
def flow_30min(interim: Path, code: str) -> pd.Series:
    f = interim / "aca_flow_qc" / f"{code}.parquet"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat qc-apply first")
    df = pd.read_parquet(f, columns=["timestamp", "value_qc"]).dropna()
    s = df.set_index("timestamp")["value_qc"].sort_index()
    return to_grid(s, "mean", FREQ)


def previous_day(daily: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Daily value of the previous complete day for each time step of the grid.

    The interval (t - 30 min, t] belongs to day D = floor(t - 30 min); it gets the value of
    day D - 1, which is already closed at time t (whether the official day is UTC or local).
    """
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).floor("D")
    d = d[~d.index.duplicated(keep="last")]
    day = (grid - pd.Timedelta(FREQ)).floor("D") - pd.Timedelta(days=1)
    return pd.Series(d.reindex(day).to_numpy(), index=grid)


def reservoir_daily(interim: Path, name: str, kind: str) -> pd.Series:
    f = interim / "aca_daily" / f"{safe_name(name)}__{kind}.parquet"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat aca-merge first")
    df = pd.read_parquet(f)
    col = "volum_embassat" if kind == "volume" and "volum_embassat" in df else "value"
    return df.set_index("timestamp")[col].astype(float)


def rain_aca_30min(interim: Path, code: str) -> pd.Series:
    f = interim / "aca_rain" / f"{code}.parquet"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat aca-rain first")
    df = pd.read_parquet(f, columns=["timestamp", "value"])
    s = df.set_index("timestamp")["value"].astype(float).sort_index()
    s = s[(s >= 0) & (s <= MAX_RAIN_5MIN_MM)]
    out = to_grid(s, "sum", FREQ)
    return out.where(out <= MAX_RAIN_30MIN_MM)


def rain_xema_30min(xema_root: Path, station: str, cfg: StationConfig) -> pd.Series:
    def one(st: str) -> pd.Series:
        obs = load_xema_history(xema_root, [st], hourly_policy=cfg.rain.hourly_policy)
        if obs.empty:
            return pd.Series(dtype=float)
        bad = obs["flag"].astype(str).str.startswith("N") | (obs["value"] < 0)
        obs = xema_period_start_to_end(obs[~bad])
        s = obs.set_index("timestamp")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return to_grid(s, "sum", FREQ)

    s = one(station)
    old = cfg.rain.splices.get(station)
    if old:
        s = splice(s, one(old)) if len(s) else one(old)
    return s.where(s <= MAX_RAIN_30MIN_MM)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def selected_inputs(interim: Path, variant: ModelVariant) -> pd.DataFrame:
    f = interim / "variants" / f"{variant.name}_inputs.csv"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run llobregat splits first")
    sel = pd.read_csv(f, encoding="utf-8-sig", dtype={"name": str})
    return sel[sel["selected"] == 1]


def assemble_wide(
    cfg: StationConfig, variant: ModelVariant, interim: Path, xema_root: Path
) -> tuple[pd.DataFrame, dict]:
    sel = selected_inputs(interim, variant)
    start = pd.Timestamp(variant.train_start, tz="UTC")
    q = flow_30min(interim, cfg.target.opendata)
    end = q.dropna().index.max()
    grid = pd.date_range(start - WARMUP, end, freq=FREQ)
    cols: dict[str, pd.Series] = {"q": q.reindex(grid)}
    inputs: dict[str, list[str]] = {}

    gauges = {g.name: g for g in cfg.upstream}
    flows = sel.loc[sel["kind"] == "flow", "name"].tolist()
    for name in flows:
        s = flow_30min(interim, gauges[name].aca.opendata).reindex(grid)
        cols[f"up:{name}"] = fill_short_gaps(s, cfg.qc.short_gap_intervals)
    inputs["flows"] = flows

    for kind, key in (("outflow", "out"), ("volume", "vol")):
        names = sel.loc[sel["kind"] == kind, "name"].tolist()
        for name in names:
            cols[f"res:{name}:{key}"] = previous_day(reservoir_daily(interim, name, kind), grid)
        inputs[f"reservoir_{kind}"] = names

    xema = sel.loc[sel["kind"] == "rain_xema", "name"].tolist()
    for st in xema:
        cols[f"rain:{st}"] = rain_xema_30min(xema_root, st, cfg).reindex(grid)
    aca = sel.loc[sel["kind"] == "rain_aca", "name"].tolist()
    for code in aca:
        cols[f"rain:{code}"] = rain_aca_30min(interim, code).reindex(grid)
    inputs["rain_xema"], inputs["rain_aca"] = xema, aca
    log.info(
        "%s: %d flows, %d reservoirs, %d rain gauges (XEMA %d, ACA %d)",
        variant.name,
        len(flows),
        len(inputs["reservoir_outflow"]),
        len(xema) + len(aca),
        len(xema),
        len(aca),
    )
    return pd.DataFrame(cols, index=grid), inputs


def build_variant_dataset(
    cfg: StationConfig, variant: ModelVariant, interim: Path, xema_root: Path
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    wide, inputs = assemble_wide(cfg, variant, interim, xema_root)
    spec = FeatureSpec.from_config(
        cfg,
        target_col="q",
        upstream_cols={n: f"up:{n}" for n in inputs["flows"]},
        rain_cols={c: f"rain:{c}" for c in inputs["rain_xema"] + inputs["rain_aca"]},
        reservoir_outflow={n: f"res:{n}:out" for n in inputs["reservoir_outflow"]},
        reservoir_volume={n: f"res:{n}:vol" for n in inputs["reservoir_volume"]},
    )
    feats = build_features(wide, spec)
    targets = {f"y_{h:g}h": wide["q"].shift(-cfg.steps(h)) for h in cfg.horizons_h}
    data = pd.concat([feats, pd.DataFrame(targets, index=wide.index)], axis=1)
    data = data[data.index >= pd.Timestamp(variant.train_start, tz="UTC")]
    data = data.astype("float32")

    # Approximate travel time (correlation of the rises), to guide the lags
    tt = []
    period = wide.index >= pd.Timestamp(variant.train_start, tz="UTC")
    for name in inputs["flows"]:
        up, down = wide.loc[period, f"up:{name}"], wide.loc[period, "q"]
        lag_all, _ = estimate_travel_time(up, down, cfg.steps(24))
        lag_hi, corr_hi = estimate_travel_time(up, down, cfg.steps(24), min_quantile=0.95)
        tt.append(
            {
                "station": name,
                "lag_h": lag_all * cfg.resolution_min / 60,
                "lag_floods_h": lag_hi * cfg.resolution_min / 60,
                "corr_floods": round(float(corr_hi.max()), 3),
            }
        )
    meta = {
        "station": cfg.station_id,
        "variant": variant.name,
        "description": variant.description,
        "start": str(data.index.min()),
        "end": str(data.index.max()),
        "rows": len(data),
        "rows_with_target": {c: int(data[c].notna().sum()) for c in targets},
        "inputs": inputs,
        "features": [c for c in feats.columns],
        "targets": list(targets),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return data, meta, pd.DataFrame(tt)


def feature_block(col: str) -> str:
    for prefix, name in (
        ("q_", "own flow"),
        ("up_", "upstream flows"),
        ("res_", "reservoirs"),
        ("rain_st_", "rain per gauge"),
        ("rain_sb_", "rain per sub-basin"),
        ("rain_basin", "basin rain"),
        ("api_", "antecedent rain"),
        ("hours_since", "antecedent rain"),
        ("doy_", "calendar"),
        ("y_", "targets"),
    ):
        if col.startswith(prefix):
            return name
    return "other"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="llobregat build-dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/stations/sant_joan_despi.yaml")
    p.add_argument("--variant", default="all")
    p.add_argument("--interim", default="data/interim")
    p.add_argument("--xema", default="data/raw/xema")
    p.add_argument("--out", default="data/processed")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_station_config(args.config)
    variants = cfg.variants if args.variant == "all" else [cfg.variant(args.variant)]
    pd.set_option("display.width", 200)
    for v in variants:
        data, meta, tt = build_variant_dataset(cfg, v, Path(args.interim), Path(args.xema))
        out = Path(args.out) / cfg.station_id / v.name
        out.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out / "dataset.parquet")
        (out / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        has_y = data["y_1h"].notna()
        nans = (data.loc[has_y].isna().mean() * 100).round(2).rename("nan_%").to_frame()
        nans["block"] = [feature_block(c) for c in nans.index]
        nans.to_csv(out / "nans.csv", encoding="utf-8-sig")
        tt.to_csv(out / "travel_times_xcorr.csv", index=False, encoding="utf-8-sig")

        print(
            f"\n=== Variant {v.name}: {meta['start'][:10]} - {meta['end'][:10]} · "
            f"{meta['rows']:,} rows · {len(meta['features'])} features ==="
        )
        print("Rows with target:", {k: f"{n:,}" for k, n in meta["rows_with_target"].items()})
        blocks = (
            nans.groupby("block")
            .agg(
                columns=("nan_%", "size"),
                nan_mean_pct=("nan_%", "mean"),
                nan_max_pct=("nan_%", "max"),
            )
            .round(2)
        )
        print(blocks.to_string())
        worst = nans[nans["block"] != "targets"].sort_values("nan_%", ascending=False).head(8)
        print("Features with most gaps (rows with target):")
        print(worst.to_string())
        if len(tt):
            print("Travel time to Sant Joan Despí (correlation of the rises):")
            print(tt.to_string(index=False))
        mb = (out / "dataset.parquet").stat().st_size / 1e6
        print(f"Saved to {out} ({mb:.0f} MB)")

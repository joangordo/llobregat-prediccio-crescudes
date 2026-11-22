"""Feature analysis, source selection, baselines and LightGBM.

Synthetic 7-year basin: the upstream flow ("castellbell") reaches the target 4 h later,
there is a station that is pure noise ("noise") and a sub-basin with a single rain gauge
(an exact duplicate of the gauge).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_basin
from llobregat import analysis, train
from llobregat.config import ModelVariant, load_station_config
from llobregat.experiment import dataset_folds, development_mask
from llobregat.features import FeatureSpec, build_features
from llobregat.selection import group_permutation_importance, select_groups

TRAVEL_STEPS = 8  # 4 h


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("processed")
    base = synthetic_basin(start="2010-10-01", years=7.0, seed=3, travel_steps=TRAVEL_STEPS)
    rng = np.random.default_rng(1)
    wide = pd.DataFrame(
        {
            "q": base["aca:P/target:flow:mean"],
            "up:castellbell": base["aca:P/up:flow:mean"],
            "up:noise": 5 + rng.normal(0, 1, len(base)),
            "rain:AA": base["xema:AA:35:sum"],
            "rain:BB": base["xema:BB:35:sum"],
            "rain:CC": base["xema:CC:35:sum"],
            "res:baells:out": 2 + 0.1 * np.sin(np.arange(len(base)) / 3000),
        },
        index=base.index,
    )
    spec = FeatureSpec(
        target_col="q",
        upstream={
            "castellbell": ("up:castellbell", [0, 1, 2, 3, 4, 6]),
            "noise": ("up:noise", [0, 1, 2]),
        },
        reservoir_outflow={"baells": "res:baells:out"},
        rain_cols={"AA": "rain:AA", "BB": "rain:BB", "CC": "rain:CC"},
        subbasins={"north": ["AA", "BB"], "south": ["CC"]},
        rain_windows_h=[1, 6, 24],
    )
    feats = build_features(wide, spec)
    targets = {f"y_{h:g}h": wide["q"].shift(-round(h * 2)) for h in (1, 6)}
    data = pd.concat([feats, pd.DataFrame(targets)], axis=1).astype("float32")

    cfg = load_station_config("configs/stations/sant_joan_despi.yaml")
    variant = ModelVariant(
        name="test_variant",
        train_start="2010-10-01",
        validation={
            "n_outer_folds": 2,
            "holdout_years": 1,
            "inner_val_years": 1,
            "calibration_years": 1,
        },
    )
    cfg = cfg.model_copy(update={"horizons_h": [1, 6], "variants": [variant]})
    out = root / cfg.station_id / "test_variant"
    out.mkdir(parents=True)
    data.to_parquet(out / "dataset.parquet")
    (out / "meta.json").write_text(json.dumps({"features": list(feats.columns)}))
    return cfg, variant, root, data


def test_feature_groups():
    g = analysis.feature_group
    assert g("q_lag_1h") == "own"
    assert g("up_anoia_sant_sadurni_lag_1p5h") == "flow:anoia_sant_sadurni"
    assert g("up_castellbell_diff_1h") == "flow:castellbell"
    assert g("res_sant_ponc_out_mean_6h") == "outflow:sant_ponc"
    assert g("res_baells_vol_diff_24h") == "volume:baells"
    assert g("rain_st_C4076a_sum_0p5h") == "rain:C4076a"
    assert g("rain_sb_alt_llobregat_imax_6h") == "subbasin:alt_llobregat"
    assert g("api_kd0p9") == "basin_rain"
    assert g("hours_since_rain") == "basin_rain"
    assert g("doy_sin") == "calendar"


def test_development_period_is_before_all_evaluation(setup):
    cfg, variant, _, data = setup
    folds = dataset_folds(data, cfg, variant)
    dev = development_mask(folds)
    first_eval = min(
        data.index[f.calib | f.test].min() for f in folds
    )  # first calibration or test time step
    assert data.index[dev].max() < first_eval
    assert [f.name for f in folds] == ["fold_1", "fold_2", "final"]


def test_flood_peaks_and_travel_time():
    t = pd.date_range("2020-01-01", periods=2000, freq="30min", tz="UTC")
    up = pd.Series(1.0, index=t)
    for c in (300, 900, 1500):
        up.iloc[c - 5 : c + 6] += np.r_[np.linspace(0, 50, 6), np.linspace(40, 0, 5)]
    down = up.shift(8).fillna(1.0)
    peaks = analysis.flood_peaks(down, 0.99, min_sep_steps=144)
    assert len(peaks) == 3
    tt = analysis.peak_travel_times(down, up, peaks, max_lag_steps=96, step_h=0.5)
    assert tt["valid"].all() and (tt["lag_h"] == 4.0).all()


def test_variables_analysis(setup):
    cfg, variant, root, _ = setup
    res = analysis.run(cfg, variant, root, sample=40_000)
    sel = res["selection"].set_index("feature")
    # The single-gauge sub-basin is an exact duplicate of the gauge
    assert sel.loc["rain_sb_south_sum_6h", "keep"] == 0
    assert sel.loc["rain_sb_south_sum_6h", "reason"].startswith("duplicate of")
    assert sel.loc["q_now", "keep"] == 1  # protected
    tt = res["travel_times"].set_index("station")
    assert abs(tt.loc["castellbell", "lag_median_h"] - 4.0) <= 0.5
    out = root / cfg.station_id / "test_variant" / "feature_analysis"
    for f in (
        "feature_selection.csv",
        "target_correlation.csv",
        "travel_times.csv",
        "summary.txt",
    ):
        assert (out / f).exists()


def test_group_permutation_drops_noise():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame(
        {
            "q_now": rng.normal(10, 2, n),
            "up_a_lag_0h": rng.normal(0, 1, n),
            "up_b_lag_0h": rng.normal(0, 1, n),
        }
    )
    y = X["q_now"] + 3 * X["up_a_lag_0h"] + rng.normal(0, 0.5, n)

    def predict(Z):
        return Z["q_now"].to_numpy() + 3 * Z["up_a_lag_0h"].to_numpy()

    imp = group_permutation_importance(
        predict, X, y, analysis.feature_groups(list(X.columns)), n_repeats=2
    )
    kept = select_groups(imp)
    assert "flow:a" in kept and "own" in kept and "flow:b" not in kept


def _args(**kw):
    base = dict(
        models=["persistence", "climatology", "ridge", "lgbm"],
        horizons=[1, 6],
        folds=None,
        features="selection",
        no_selection=False,
        transform="log_ratio",
        ridge_transform="delta",
        tweedie=False,
        weights="uniform",
        rapid=True,
        linear_trees=False,
        max_rounds=150,
        name=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_train_and_compare(setup):
    cfg, variant, root, _ = setup
    sel_file = root / cfg.station_id / "test_variant" / "feature_analysis" / "feature_selection.csv"
    if not sel_file.exists():
        analysis.run(cfg, variant, root, sample=40_000)
    written = train.run(cfg, variant, root, _args())
    assert set(written) == {"persistence", "climatology", "ridge", "lgbm"}
    met = pd.concat(pd.read_csv(d / "metrics.csv") for d in written.values())
    by = met.groupby(["model", "horizon_h"])["pi"].mean()
    assert abs(by["persistence"]).max() < 1e-9  # persistence PI = 0
    # With a 4 h travel time, LightGBM must clearly beat persistence
    assert by["lgbm"].min() > 0.2
    assert by["ridge"].min() > 0
    # Same rows for every model
    files = met.groupby(["fold", "horizon_h"])["rows"].nunique()
    assert (files == 1).all()
    # No forecast of the hold-out (final) or of any non-test year
    pred = pd.read_parquet(written["lgbm"] / "predictions.parquet")
    assert set(pred["fold"]) == {"fold_1", "fold_2"}
    assert set(pred["hydro_year"]) == {2015, 2016}
    # The noise source is dropped in most cases
    sources = pd.read_csv(written["lgbm"] / "source_importance.csv")
    noise = sources[sources["group"] == "flow:noise"]["kept"].mean()
    signal = sources[sources["group"] == "flow:castellbell"]["kept"].mean()
    assert signal == 1.0 and noise < 0.5

    summary = train.compare(cfg, variant, root, n_boot=50)
    assert {"lgbm", "ridge", "persistence", "climatology"} <= set(summary["model"])
    stats = pd.read_csv(root / cfg.station_id / "test_variant" / "models" / "comparison_stats.csv")
    row = stats[(stats["model"] == "lgbm") & (stats["reference"] == "persistence")]
    assert (row["nse_diff_ci95_low"] > 0).all()


def test_model_names():
    assert train.model_name("lgbm", _args()) == "lgbm"
    assert train.model_name("lgbm", _args(features="all", no_selection=True)) == (
        "lgbm_allfeatures_noselection"
    )
    assert (
        train.model_name("lgbm", _args(weights="tfg_stepped", name="v2")) == "lgbm_tfg_stepped_v2"
    )
    assert train.model_name("persistence", _args(features="all")) == "persistence"
    tw = _args(transform="level", tweedie=True)
    assert train.model_name("lgbm", tw) == "lgbm_level_tweedie"


def test_peak_error_of_persistence_is_zero_at_long_horizons():
    t = pd.date_range("2020-01-01", periods=600, freq="30min", tz="UTC")
    q = pd.Series(10.0, index=t)
    q.iloc[200:230] = np.r_[np.linspace(10, 300, 15), np.linspace(280, 10, 15)]
    h = 24
    steps = h * 2
    pred = pd.DataFrame({"obs": q.shift(-steps), "pred": q, "q_now": q}, index=t).dropna()
    m = train.event_metrics(pred, threshold=100, h=h, step_h=0.5)
    assert m["episodes"] == 1
    assert abs(m["peak_error_pct"]) < 1e-6 and m["peak_timing_error_h"] == 24.0


def test_compare_variants_uses_common_years(setup):
    cfg, variant, root, data = setup
    v2 = ModelVariant(
        name="test_variant2",
        train_start="2011-10-01",
        validation={
            "n_outer_folds": 1,
            "holdout_years": 1,
            "inner_val_years": 1,
            "calibration_years": 1,
        },
    )
    cfg2 = cfg.model_copy(update={"variants": [variant, v2]})
    out = root / cfg.station_id / "test_variant2"
    out.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out / "dataset.parquet")
    args = _args(models=["persistence", "lgbm"], features="all", horizons=[1])
    train.run(cfg2, v2, root, args)
    train.run(cfg2, variant, root, args)  # same model name (lgbm_allfeatures) in both
    summary = train.compare_variants(cfg2, root, n_boot=20)
    assert set(summary["variant"]) == {"test_variant", "test_variant2"}
    text = (root / cfg.station_id / "variant_summary.txt").read_text(encoding="utf-8")
    assert "(2016)" in text  # the only shared test year
    stats = pd.read_csv(root / cfg.station_id / "variant_comparison_stats.csv")
    assert set(stats["model"]) == {"lgbm_allfeatures"}
    assert set(stats["rows"]) == {stats["rows"].iloc[0]}  # same rows in both


def test_sample_weights():
    y = np.r_[np.ones(200), np.linspace(1, 50, 20), np.linspace(50, 60, 30), np.ones(250)]
    for s in train.WEIGHTS:
        w = train.sample_weights(s, y, h=6, step_h=0.5)
        assert w.shape == y.shape and abs(w.mean() - 1) < 1e-9 and (w > 0).all()
    w = train.sample_weights("episodes", y, h=6, step_h=0.5)
    # the rise (up to 6 h before exceeding p95) also weighs more
    first_high = int(np.argmax(y > np.quantile(y, 0.95)))
    assert w[first_high - 12] > w[0]
    assert train.sample_weights("uniform", y, 6, 0.5).std() == 0


def test_pooled_metrics_and_linear_trees(setup):
    cfg, variant, root, _ = setup
    args = _args(models=["persistence", "lgbm"], features="all", horizons=[6], linear_trees=True)
    assert train.model_name("lgbm", args) == "lgbm_allfeatures_linear"
    written = train.run(cfg, variant, root, args)
    pred = pd.read_parquet(written["lgbm_allfeatures_linear"] / "predictions.parquet")
    met = pd.read_csv(written["lgbm_allfeatures_linear"] / "metrics.csv")
    pooled = train.pooled_metrics(pred, met, step_h=0.5)
    assert len(pooled) == 1 and pooled["rows"].iloc[0] == len(pred)
    # The pooled NSE is that of all rows together, not the mean of the folds
    from llobregat.metrics import nse

    assert abs(pooled["nse"].iloc[0] - nse(pred["obs"], pred["pred"])) < 1e-9
    assert pooled["episodes"].iloc[0] == met["episodes"].sum()
    assert pooled["pi"].iloc[0] > 0


def test_combine_models(setup):
    cfg, variant, root, _ = setup
    a = _args(models=["persistence", "ridge", "lgbm"], features="all", horizons=[1], name="c")
    train.run(cfg, variant, root, a)
    models = ["lgbm_allfeatures_c", "ridge_allfeatures_c"]
    out = train.combine(cfg, variant, root, models, name="mean")
    pred = pd.read_parquet(out / "predictions.parquet")
    mdir = root / cfg.station_id / "test_variant" / "models"
    pa = pd.read_parquet(mdir / "lgbm_allfeatures_c" / "predictions.parquet")
    pb = pd.read_parquet(mdir / "ridge_allfeatures_c" / "predictions.parquet")
    assert len(pred) == len(pa)
    expected = (pa["pred"].astype(float) + pb.loc[pa.index, "pred"].astype(float)) / 2
    assert np.allclose(pred["pred"].to_numpy(), expected.to_numpy(), atol=1e-3)
    met = pd.read_csv(out / "metrics.csv")
    assert set(met["fold"]) == {"fold_1", "fold_2"} and met["nse"].notna().all()


def test_tuning_and_training_with_tuned_params(setup):
    from llobregat import tuning

    cfg, variant, root, _ = setup
    table = tuning.tune(cfg, variant, root, [1], n_trials=3, max_rounds=60, features="all")
    assert len(table) == 1 and table["rmse_tuned"].iloc[0] <= table["rmse_default"].iloc[0]
    d = tuning.params_dir(root, cfg, variant)
    params = tuning.load_tuned(d, 1)
    assert set(params) == set(tuning.TUNED_KEYS)
    args = _args(models=["lgbm"], features="all", horizons=[1], params="tuned")
    args.params_dir = d
    assert train.model_name("lgbm", args) == "lgbm_allfeatures_tuned"
    assert train.lgbm_params(args, 1)["num_leaves"] == params["num_leaves"]
    written = train.run(cfg, variant, root, args)
    cfg_run = json.loads((written["lgbm_allfeatures_tuned"] / "config.json").read_text())
    assert cfg_run["lgbm_params"]["1h"]["num_leaves"] == params["num_leaves"]


def test_final_params_keep_defaults_without_clear_gain(tmp_path):
    from llobregat import tuning

    for h, (mse_t, mse_d) in {1: (100.0, 100.0), 3: (64.0, 100.0)}.items():
        (tmp_path / f"params_{h}h.json").write_text(
            json.dumps(
                {
                    "params": {"num_leaves": 200},
                    "mse_validation": mse_t,
                    "mse_validation_default": mse_d,
                }
            )
        )
    out = tuning.final_params(tmp_path, [1, 3], min_gain_pct=5)
    assert tuning.load_tuned(out, 1) == tuning.default_trial()  # 0 %: defaults
    assert tuning.load_tuned(out, 3) == {"num_leaves": 200}  # 20 %: tuned

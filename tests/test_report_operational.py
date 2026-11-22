"""Final report and operational model (synthetic basin of test_block3)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from llobregat import analysis, operational, report
from llobregat import probabilistic as P
from llobregat.experiment import dataset_dir, dataset_folds
from test_block3 import setup  # noqa: F401  (shared fixture)


def test_lead_times_counts_only_warnings_issued_below_threshold():
    t = pd.date_range("2020-01-01", periods=200, freq="30min", tz="UTC")
    q = pd.Series(10.0, index=t)
    q.iloc[100:120] = 200  # episode starts at step 100
    rows = []
    for h in (1, 3):
        steps = round(h * 2)
        df = pd.DataFrame({"q_now": q.to_numpy(), "obs": q.shift(-steps).to_numpy()}, index=t)
        df["central"] = df["obs"]  # perfect forecast
        df["hi90_cqr"] = df["obs"] * 1.5
        rows.append(df.assign(horizon_h=h))
    pred = pd.concat(rows)
    lt = report.lead_times(pred, threshold=100.0)
    assert len(lt) == 1
    assert lt["alert_lead_h"].iloc[0] == pytest.approx(3.0)  # the 3 h forecast is earliest
    assert lt["alert_horizon_h"].iloc[0] == 3


def test_operational_model_and_report(setup):  # noqa: F811
    cfg, variant, root, data = setup
    analysis.run(cfg, variant, root, sample=40_000)
    base = dataset_dir(root, cfg, variant)
    # Masks: training before the last complete year, calibration on it, no test
    fm = operational.operational_masks(data, cfg, variant)
    assert not fm.test.any() and fm.calib.any()
    assert data.index[fm.train].max() < data.index[fm.calib].min()

    out = operational.train_operational(cfg, variant, root, [1], max_rounds=60, rapid=True)
    man = json.loads((out / "manifest.json").read_text())
    assert man["calibration_years"] == [int(y) for y in fm.fold.calib_years]
    model = operational.OperationalModel.load(out)
    X = data.loc[fm.calib].iloc[:500]
    fc = model.predict(X)
    assert len(fc) == 500 and set(fc["horizon_h"]) == {1.0}
    assert (fc["lo80"] <= fc["hi80"]).all() and (fc["lo90"] <= fc["hi90"]).all()
    # The two levels are calibrated separately, so they are nested almost always (not always)
    assert (fc["hi80"] <= fc["hi90"] + 1e-6).mean() > 0.8
    assert (fc["pre_alert"] | ~fc["alert"]).mean() > 0.95  # alert almost implies pre-alert
    assert (out / "source_importance.csv").exists()

    # Report from the outputs of the probabilistic runs
    folds = [f for f in dataset_folds(data, cfg, variant) if f.name != "final"]
    P.run(cfg, variant, root, folds, [1], base / "uncertainty", max_rounds=60, rapid=True)
    P.final_evaluation(cfg, variant, root, [1], max_rounds=60, rapid=True, force=True)
    rep = report.run(cfg, variant, root)
    pm = pd.read_csv(rep / "point_metrics.csv")
    assert {"hold-out", "cross-validation"} <= set(pm["set"])
    assert np.isfinite(pm["nse"]).all()
    assert (rep / "figures" / "skill_by_horizon.png").exists()
    assert "Skill by lead time" in (rep / "report.md").read_text()

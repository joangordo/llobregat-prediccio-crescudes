"""The migration script turns the Catalan output layout into the English one."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "migrate", Path(__file__).parents[1] / "scripts" / "migrate_outputs_to_english.py"
)
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)


def test_migration_renames_folders_files_columns_and_keys(tmp_path):
    root = tmp_path / "data"
    v = root / "processed" / "sant_joan_despi" / "llarg"
    (v / "incertesa").mkdir(parents=True)
    (v / "final").mkdir()
    (v / "variables").mkdir()
    (v / "ajust" / "final").mkdir(parents=True)
    (v / "models" / "lgbm_totes_ajustat").mkdir(parents=True)
    t = pd.date_range("2025-01-01", periods=3, freq="30min", tz="UTC")
    pd.DataFrame(
        {
            "obs": 1.0,
            "plec": "final",
            "horitzo_h": 1.0,
            "any_hidrologic": 2025,
            "lo80_quantils": 0.5,
            "hi90_persistencia": 2.0,
            "sup90": 1.5,
        },
        index=t,
    ).to_parquet(v / "final" / "prediccions.parquet")
    pd.DataFrame({"plec": ["final"], "regla": ["sup90"], "llindar": [450.0]}).to_csv(
        v / "final" / "avisos.csv", index=False
    )
    (v / "final" / "AVALUACIO_FETA.json").write_text(
        json.dumps({"data": "2026-11-20", "empremta": "abc", "historial": []})
    )
    pd.DataFrame(
        {"grup": ["propi", "cabal:castellbell"], "augment_mse": [1.0, 0.2], "horitzo_h": 1}
    ).to_csv(v / "models" / "lgbm_totes_ajustat" / "importancia_fonts.csv", index=False)
    pd.DataFrame({"variable": ["q_now"], "conserva": [1], "motiu": ["duplicat de x"]}).to_csv(
        v / "variables" / "seleccio_variables.csv", index=False
    )
    (v / "ajust" / "final" / "parametres_3h.json").write_text(
        json.dumps({"horitzo_h": 3, "parametres": {"num_leaves": 31}, "font": "ajustats"})
    )
    qc = root / "interim" / "aca_cabals_qc"
    qc.mkdir(parents=True)
    pd.DataFrame(
        {"timestamp": t, "value": 1.0, "origen": "cabals", "flags": ["", "pic,caiguda", ""]}
    ).to_parquet(qc / "1.parquet")
    pd.DataFrame({"nom": ["a"], "pic": [3]}).to_csv(qc / "resum.csv", index=False)

    migrate.main(["--data", str(root), "--apply"])
    v2 = root / "processed" / "sant_joan_despi" / "long"
    pred = pd.read_parquet(v2 / "final" / "predictions.parquet")
    assert {
        "fold",
        "horizon_h",
        "hydro_year",
        "lo80_quantiles",
        "hi90_persistence",
        "upper90",
    } <= set(pred.columns)
    w = pd.read_csv(v2 / "final" / "warnings.csv")
    assert w["rule"].tolist() == ["upper90"] and "threshold" in w
    lock = json.loads((v2 / "final" / "HOLDOUT_EVALUATED.json").read_text())
    assert lock["fingerprint"] == "abc" and "migrated" in lock
    imp = pd.read_csv(v2 / "models" / "lgbm_allfeatures_tuned" / "source_importance.csv")
    assert imp["group"].tolist() == ["own", "flow:castellbell"]
    sel = pd.read_csv(v2 / "feature_analysis" / "feature_selection.csv")
    assert (
        sel.columns.tolist() == ["feature", "keep", "reason"]
        and sel["reason"][0] == "duplicate of x"
    )
    par = json.loads((v2 / "tuning" / "final" / "params_3h.json").read_text())
    assert par == {"horizon_h": 3, "params": {"num_leaves": 31}, "source": "tuned"}
    q = pd.read_parquet(root / "interim" / "aca_flow_qc" / "1.parquet")
    assert q["flags"].tolist() == ["", "spike,dropout", ""] and "source" in q
    assert pd.read_csv(root / "interim" / "aca_flow_qc" / "summary.csv").columns.tolist() == [
        "name",
        "spike",
    ]
    # Idempotent
    migrate.main(["--data", str(root), "--apply"])
    assert (v2 / "final" / "predictions.parquet").exists()

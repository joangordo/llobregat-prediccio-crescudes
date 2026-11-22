"""Rename the data and outputs produced by the Catalan version of the code to English names.

Up to November 2026 the code wrote Catalan folder, file and column names (llarg/,
incertesa/, prediccions.parquet, plec, horitzo_h...). The code now uses English names. This
script renames an existing data/ folder in place so that nothing has to be re-run (in
particular, the final hold-out evaluation does not have to be repeated).

    uv run python scripts/migrate_outputs_to_english.py            # dry run: list changes
    uv run python scripts/migrate_outputs_to_english.py --apply    # do it

It is idempotent: running it twice changes nothing the second time. dataset.parquet and the
raw downloads are not modified (their columns are already English or are source names).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Folders and files
# --------------------------------------------------------------------------- #
DIRS = {
    "aca_cabals": "aca_flow",
    "aca_cabals_qc": "aca_flow_qc",
    "aca_diaris": "aca_daily",
    "aca_pluja": "aca_rain",
    "llarg": "long",
    "complet": "full",
    "variables": "feature_analysis",
    "incertesa": "uncertainty",
    "ajust": "tuning",
}
FILES = {
    "prediccions.parquet": "predictions.parquet",
    "metriques.csv": "metrics.csv",
    "metriques_totes.csv": "metrics_all.csv",
    "avisos.csv": "warnings.csv",
    "resum.txt": "summary.txt",
    "resum.csv": "summary.csv",
    "importancia_fonts.csv": "source_importance.csv",
    "importancia_variables.csv": "feature_importance.csv",
    "comparacio.csv": "comparison.csv",
    "comparacio_mitjana_plecs.csv": "comparison_fold_mean.csv",
    "comparacio_estadistica.csv": "comparison_stats.csv",
    "comparacio_variants.csv": "variant_comparison.csv",
    "comparacio_variants_estadistica.csv": "variant_comparison_stats.csv",
    "resum_variants.txt": "variant_summary.txt",
    "comparacio_tfg.csv": "tfg_comparison.csv",
    "AVALUACIO_FETA.json": "HOLDOUT_EVALUATED.json",
    "seleccio_variables.csv": "feature_selection.csv",
    "correlacio_objectiu.csv": "target_correlation.csv",
    "duplicats.csv": "duplicates.csv",
    "correlacions.parquet": "correlations.parquet",
    "temps_viatge.csv": "travel_times.csv",
    "temps_viatge_pics.csv": "travel_time_peaks.csv",
    "cobertura.csv": "coverage.csv",
    "qualitat.csv": "quality.csv",
    "unitats.csv": "units.csv",
    "dies_balanc.csv": "balance_days.csv",
    "informe.csv": "report.csv",
    "pics.csv": "peaks.csv",
    "maxims.txt": "maxima.txt",
    "maxims_nets.txt": "maxima_clean.txt",
    "estacions.csv": "stations.csv",
    "cataleg_aca.csv": "aca_catalog.csv",
}
FILE_PATTERNS = [
    (r"^parametres_(.+)\.json$", r"params_\1.json"),
    (r"^proves_(.+)\.csv$", r"trials_\1.csv"),
    (r"^(.+)_entrades\.csv$", r"\1_inputs.csv"),
    (r"^(.+)_plecs\.csv$", r"\1_folds.csv"),
    (r"^(.+)__cabal\.parquet$", r"\1__flow.parquet"),
    (r"^(.+)__sortida\.parquet$", r"\1__outflow.parquet"),
    (r"^(.+)__volum\.parquet$", r"\1__volume.parquet"),
    (r"^(.+)__nivell\.parquet$", r"\1__level.parquet"),
]
VARIANTS = {"llarg": "long", "complet": "full"}
MODEL_TOKENS = {
    "persistencia": "persistence",
    "climatologia": "climatology",
    "totes": "allfeatures",
    "sensesel": "noselection",
    "potencia": "power",
    "episodis": "episodes",
    "lineal": "linear",
    "ajustat": "tuned",
}

# --------------------------------------------------------------------------- #
# Columns and values
# --------------------------------------------------------------------------- #
COLUMNS = {
    # predictions, metrics, comparisons
    "plec": "fold",
    "horitzo_h": "horizon_h",
    "any_hidrologic": "hydro_year",
    "any_test": "test_year",
    "files": "rows",
    "segons": "seconds",
    "iteracions": "iterations",
    "llindar_avis": "warning_threshold",
    "llindar_alt": "high_threshold",
    "rmse_alts": "rmse_high",
    "biaix_alts_pct": "bias_high_pct",
    "pi_alts": "pi_high",
    "episodis": "episodes",
    "encerts": "hits",
    "falses_alarmes": "false_alarms",
    "no_detectats": "misses",
    "error_pic_pct": "peak_error_pct",
    "error_pic_abs_pct": "peak_abs_error_pct",
    "error_temps_pic_h": "peak_timing_error_h",
    "conservada": "kept",
    "referencia": "reference",
    "nse_referencia": "nse_reference",
    "dif_nse": "nse_diff",
    "dif_nse_ic95_baix": "nse_diff_ci95_low",
    "dif_nse_ic95_alt": "nse_diff_ci95_high",
    "dm_estadistic": "dm_stat",
    # intervals and warnings
    "metode": "method",
    "nivell": "level",
    "cobertura": "coverage",
    "cobertura_alts": "coverage_high",
    "per_sota_pct": "below_pct",
    "per_sobre_pct": "above_pct",
    "amplada_m3s": "width_m3s",
    "amplada_mediana_rel": "median_rel_width",
    "amplada_alts_m3s": "width_high_m3s",
    "regla": "rule",
    "triada": "chosen",
    "llindar": "threshold",
    "sup80": "upper80",
    "sup90": "upper90",
    # importance and feature analysis
    "grup": "group",
    "augment_mse": "mse_increase",
    "augment_mse_min": "mse_increase_min",
    "guany": "gain",
    "divisions": "splits",
    "guany_pct": "gain_pct",
    "bloc": "block",
    "conserva": "keep",
    "motiu": "reason",
    "grup_duplicats": "duplicate_group",
    "rho_canvi_abs_mitja": "rho_change_abs_mean",
    # travel times
    "estacio": "station",
    "crescudes": "floods",
    "valides": "valid",
    "retard_mediana_h": "lag_median_h",
    "retard_p25_h": "lag_p25_h",
    "retard_p75_h": "lag_p75_h",
    "retard_correlacio_h": "lag_correlation_h",
    "cabal_pic": "peak_flow",
    "pic_dalt": "upstream_peak",
    "cabal_pic_dalt": "upstream_peak_flow",
    "retard_h": "lag_h",
    "retard_crescudes_h": "lag_floods_h",
    "corr_crescudes": "corr_floods",
    # tuning
    "rmse_per_defecte": "rmse_default",
    "rmse_ajustat": "rmse_tuned",
    "millora_pct": "gain_pct",
    "user_attrs_iteracions": "user_attrs_iterations",
    "user_attrs_segons": "user_attrs_seconds",
    # interim data
    "nivell_qualitat": "quality_level",
    "origen": "source",
    "value_brut": "value_raw",
    "factor_unitat": "unit_factor",
    "nom": "name",
    "tipus": "kind",
    "entra": "selected",
    "inici": "start",
    "fi": "end",
    "dies": "days",
    "lectures_ls_%": "readings_ls_%",
    "anys_majoria_ls": "years_mostly_ls",
    "lectures": "readings",
    "marcades_%": "flagged_%",
    "dies_balanc": "balance_days",
    "max_abans": "max_before",
    "max_net": "max_clean",
    "data_max_net": "max_clean_time",
    "mitjana_5min": "mean_5min",
    "diari_oficial": "official_daily",
    "quocient": "ratio",
    "negatius": "negatives",
    "data_max": "max_time",
    "salt30_p99_9": "jump30_p99_9",
    "salt30_max": "jump30_max",
    "data_salt_max": "jump_max_time",
    "pics_aillats": "isolated_spikes",
    "trams_plans_llargs": "long_flat_runs",
    "hores_tram_pla_max": "longest_flat_run_h",
    "ordre": "rank",
    "cabal_30min": "flow_30min",
    "cobertura_des_de_regular": "coverage_since_regular",
    "pluja_anual_mediana_mm": "median_annual_rain_mm",
    # quality-control rules (summary columns and flags)
    "excursio": "excursion",
    "caiguda": "dropout",
    "oscil·lacio": "oscillation",
    "encallat": "stuck",
    "pla_alt": "high_flat",
    "pas_unitat": "unit_step",
    "balanc_diari": "daily_balance",
    "exclusio": "exclusion",
}
# Columns that mean something else in the interim data (long-format observations)
PROCESSED_ONLY = {"variables": "n_features", "variable": "feature", "pic": "peak"}
INTERIM_ONLY = {"pic": "spike", "data": "time", "dia": "day", "codi": "code", "conca": "basin"}
COLUMN_PATTERNS = [
    (r"^rho_canvi_(.+)$", r"rho_change_\1"),
    (r"^rho_nivell_(.+)$", r"rho_level_\1"),
    (r"^(lo|hi)(\d+)_quantils$", r"\1\2_quantiles"),
    (r"^(lo|hi)(\d+)_persistencia$", r"\1\2_persistence"),
    (r"^dies_(.+)_%$", r"days_\1_%"),
    (r"^(NQ_REV\d)_pics_per_100k$", r"\1_spikes_per_100k"),
    (r"^(NQ_REV\d)_pla_%$", r"\1_flat_%"),
]
GROUP_PREFIX = {
    "cabal:": "flow:",
    "sortida:": "outflow:",
    "volum:": "volume:",
    "pluja:": "rain:",
    "subconca:": "subbasin:",
}
VALUES = {
    "method": {"quantils": "quantiles", "persistencia": "persistence"},
    "rule": {"sup80": "upper80", "sup90": "upper90", "triada per plec": "chosen per fold"},
    "group": {
        "propi": "own",
        "calendari": "calendar",
        "pluja_conca": "basin_rain",
        "altres": "other",
    },
    "block": {
        "cabal propi": "own flow",
        "cabals aigües amunt": "upstream flows",
        "embassaments": "reservoirs",
        "pluja per estació": "rain per gauge",
        "pluja per subconca": "rain per sub-basin",
        "pluja conca": "basin rain",
        "pluja antecedent": "antecedent rain",
        "calendari": "calendar",
        "objectius": "targets",
        "altres": "other",
    },
    "kind": {
        "cabal": "flow",
        "sortida": "outflow",
        "volum": "volume",
        "nivell": "level",
        "pluja_xema": "rain_xema",
        "pluja_aca": "rain_aca",
    },
    "source": {"aca_pluja": "aca_rain"},
    "variant": VARIANTS,
    "variant_a": VARIANTS,
    "variant_b": VARIANTS,
    "class": {"sense_ref": "no_ref", "mixt": "mixed", "estrany": "odd"},
}
FLAG_RULES = {
    "excursio": "excursion",
    "pic": "spike",
    "caiguda": "dropout",
    "oscil·lacio": "oscillation",
    "encallat": "stuck",
    "pla_alt": "high_flat",
    "pas_unitat": "unit_step",
    "balanc_diari": "daily_balance",
    "exclusio": "exclusion",
}
JSON_KEYS = {
    # meta.json
    "estacio": "station",
    "descripcio": "description",
    "inici": "start",
    "fi": "end",
    "files": "rows",
    "files_amb_objectiu": "rows_with_target",
    "entrades": "inputs",
    "variables": "features",
    "objectius": "targets",
    "creat": "created",
    "cabals": "flows",
    "embassaments_sortida": "reservoir_outflow",
    "embassaments_volum": "reservoir_volume",
    "pluja_xema": "rain_xema",
    "pluja_aca": "rain_aca",
    # model config.json
    "plecs": "folds",
    "anys_test": "test_years",
    "horitzons_h": "horizons_h",
    "n_variables": "n_features",
    "transformacio": "transform",
    "seleccio_fonts": "source_selection",
    "pesos": "weights",
    "parametres_lgbm": "lgbm_params",
    "combinacio": "combination",
    # tuning
    "horitzo_h": "horizon_h",
    "parametres": "params",
    "mse_validacio": "mse_validation",
    "mse_validacio_per_defecte": "mse_validation_default",
    "iteracions": "iterations",
    "anys_validacio": "validation_years",
    "proves": "trials",
    "font": "source",
    "millora_validacio_pct": "validation_gain_pct",
    "llindar_pct": "threshold_pct",
    # hold-out lock
    "data": "date",
    "empremta": "fingerprint",
    "historial": "history",
}
JSON_VALUES = {
    "ajustats": "tuned",
    "per defecte": "default",
    "seleccio": "selection",
    "totes": "all",
    **VARIANTS,
}
NAME_FIXES = [("(objectiu)", "(target)"), (" · cabal", " · flow"), (" · nivell", " · level")]


def model_name(name: str) -> str:
    parts = re.split(r"([_+])", name)
    return "".join(MODEL_TOKENS.get(p, p) for p in parts)


def rename_column(c: str, processed: bool) -> str:
    if not isinstance(c, str):
        return c
    if processed and c in PROCESSED_ONLY:
        return PROCESSED_ONLY[c]
    if not processed and c in INTERIM_ONLY:
        return INTERIM_ONLY[c]
    if c in COLUMNS:
        return COLUMNS[c]
    for pat, rep in COLUMN_PATTERNS:
        if re.match(pat, c):
            return re.sub(pat, rep, c)
    return c


def translate_value(col: str, v):
    if not isinstance(v, str):
        return v
    if col in ("model", "reference"):
        return model_name(v)
    if col == "model_variant":
        m = re.match(r"^(.*) \((.*)\)$", v)
        return f"{model_name(m[1])} ({VARIANTS.get(m[2], m[2])})" if m else v
    if col == "group":
        for a, b in GROUP_PREFIX.items():
            if v.startswith(a):
                return b + v[len(a) :]
        return VALUES["group"].get(v, v)
    if col == "reason":
        if v == "massa forats":
            return "too many gaps"
        return re.sub(r"^duplicat de ", "duplicate of ", v)
    if col == "flags":
        return ",".join(FLAG_RULES.get(x, x) for x in v.split(",")) if v else v
    if col == "name":
        for a, b in NAME_FIXES:
            v = v.replace(a, b)
        return v
    return VALUES.get(col, {}).get(v, v)


def migrate_table(df: pd.DataFrame, processed: bool) -> tuple[pd.DataFrame, bool]:
    cols = {c: rename_column(c, processed) for c in df.columns}
    out = df.rename(columns=cols)
    changed = any(a != b for a, b in cols.items())
    text_cols = {*VALUES, "model", "reference", "model_variant", "group", "reason", "flags", "name"}
    for col in out.columns:
        if col in text_cols and not pd.api.types.is_numeric_dtype(out[col]):
            old = out[col].astype(object)
            new = old.map(lambda v, c=col: translate_value(c, v))
            if not new.equals(old):
                out[col] = new
                changed = True
    return out, changed


def migrate_json(obj):
    if isinstance(obj, dict):
        return {JSON_KEYS.get(k, k): migrate_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [migrate_json(v) for v in obj]
    if isinstance(obj, str):
        return JSON_VALUES.get(obj, obj)
    return obj


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def new_name(path: Path, is_dir: bool, parent_is_models: bool) -> str:
    name = path.name
    if is_dir:
        if parent_is_models:
            return model_name(name)
        return DIRS.get(name, name)
    if name in FILES:
        return FILES[name]
    for pat, rep in FILE_PATTERNS:
        if re.match(pat, name):
            name = re.sub(pat, rep, name)
            break
    for a, b in VARIANTS.items():
        if name.startswith(a + "_") and name.endswith((".csv",)):
            name = b + name[len(a) :]
    return name


def rename_tree(root: Path, apply: bool, log: list[str]) -> None:
    # Deepest first, so that renaming a folder does not invalidate the paths below it
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if "raw" in p.relative_to(root).parts[:1] and p.is_dir() and p.name != "aca_pluja":
            continue
        target = p.with_name(new_name(p, p.is_dir(), p.parent.name == "models"))
        if target != p:
            log.append(f"rename {p.relative_to(root)} -> {target.name}")
            if apply:
                if target.exists():
                    raise SystemExit(f"{target} already exists: resolve it by hand")
                p.rename(target)


def rewrite_contents(root: Path, apply: bool, log: list[str]) -> None:
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or rel.parts[0] == "raw" or p.name == "dataset.parquet":
            continue
        processed = rel.parts[0] == "processed"
        try:
            if p.suffix == ".csv":
                df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
                out, changed = migrate_table(df, processed)
                if changed:
                    log.append(f"columns {rel}")
                    if apply:
                        out.to_csv(p, index=False, encoding="utf-8-sig")
            elif p.suffix == ".parquet" and p.name != "correlations.parquet":
                df = pd.read_parquet(p)
                out, changed = migrate_table(df, processed)
                if changed:
                    log.append(f"columns {rel}")
                    if apply:
                        out.to_parquet(p)
            elif p.suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                new = migrate_json(obj)
                if p.name == "config.json" and isinstance(new, dict) and "model" in new:
                    new["model"] = model_name(new["model"])
                    if isinstance(new.get("combination"), dict):
                        new["combination"] = {
                            model_name(k): v for k, v in new["combination"].items()
                        }
                if p.name == "HOLDOUT_EVALUATED.json" and "migrated" not in new:
                    new["migrated"] = datetime.now(UTC).isoformat(timespec="seconds")
                if new != obj:
                    log.append(f"keys {rel}")
                    if apply:
                        p.write_text(json.dumps(new, ensure_ascii=False, indent=2), "utf-8")
        except Exception as exc:
            log.append(f"SKIPPED {rel}: {exc}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="data", help="The data folder to migrate")
    ap.add_argument("--apply", action="store_true", help="Make the changes (default: dry run)")
    args = ap.parse_args(argv)
    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    log: list[str] = []
    rename_tree(root, args.apply, log)
    if args.apply:
        rewrite_contents(root, True, log)
    else:
        log.append("(column and key changes are listed with --apply)")
    print("\n".join(log) if log else "Nothing to migrate")
    print(f"\n{len(log)} changes" + ("" if args.apply else " (dry run; use --apply)"))


if __name__ == "__main__":
    main()

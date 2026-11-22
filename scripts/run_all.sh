#!/usr/bin/env bash
# Rebuild everything from the raw data to the report and the operational model.
#
# Prerequisites (data are not in the repository, see docs/reproduir.md):
#   data/raw/aca_opendata/   ACA open data (llobregat aca-opendata ..., or download_aca_history.bat)
#   data/raw/aca_rain/       ACA rain gauge Excel files (supplied by the ACA on request)
#   data/raw/xema/           XEMA open data (llobregat xema ...), only for the "full" variant
#
# The final evaluation on the hold-out is locked after its first run. Running this script
# again stops there unless FORCE_HOLDOUT=1 is set, and a forced run is recorded in
# data/processed/<station>/<variant>/final/HOLDOUT_EVALUATED.json.
set -euo pipefail
cd "$(dirname "$0")/.."
V=${VARIANT:-long}
RUN="uv run --frozen --all-extras llobregat"

$RUN aca-merge
$RUN aca-rain --basins Llobregat
$RUN qc-apply
$RUN splits
$RUN build-dataset --variant "$V"
$RUN analyse-features --variant "$V"

# Point forecasts: baselines, Ridge, LightGBM, tuning and the fixed-weight ensemble
$RUN train --variant "$V" --models persistence climatology ridge
$RUN tune --variant "$V"
$RUN tune --variant "$V" --final
$RUN train --variant "$V" --models lgbm --params "data/processed/sant_joan_despi/$V/tuning/final"
$RUN combine --variant "$V" --models lgbm_tuned ridge --name ensemble
$RUN compare --variant "$V"

# Probabilistic forecasts and warnings (cross-validation), then the hold-out, once
$RUN intervals --variant "$V"
if [[ "${FORCE_HOLDOUT:-0}" == "1" ]]; then
  $RUN final-evaluation --variant "$V" --force
else
  $RUN final-evaluation --variant "$V" || echo "Hold-out already evaluated: kept as it is"
fi

# Report figures and tables, and the operational model used by the real-time app
$RUN report --variant "$V"
$RUN operational-model --variant "$V"

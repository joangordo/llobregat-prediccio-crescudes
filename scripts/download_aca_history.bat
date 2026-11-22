@echo off
rem Download the ACA history from the open data portal (flows 1996-2019, daily flows since
rem 1951 and daily reservoir status since 2000). It can be run again: it resumes where it
rem stopped. Keep the computer from going to sleep while it runs.
chcp 65001 >nul
cd /d "%~dp0.."
set S=data\raw\aca_opendata

echo === Flows 2010-2019 (5 min) ===
python -m uv run llobregat aca-opendata data --dataset cabals_2010_2019 --stations-file %S%\stations_cabals.csv
echo === Flows 1996-2009 (5 min) ===
python -m uv run llobregat aca-opendata data --dataset cabals_1996_2009 --stations-file %S%\stations_cabals.csv
echo === Daily flows (including dam outflows) ===
python -m uv run llobregat aca-opendata data --dataset cabals_diaris --stations-file %S%\stations_cabals_diaris.csv
echo === Daily reservoir status (Baells, Llosa del Cavall, Sant Ponc) ===
python -m uv run llobregat aca-opendata stations --dataset embassaments_diaris --name Baells Llosa "Sant Pon"
python -m uv run llobregat aca-opendata data --dataset embassaments_diaris --stations-file %S%\stations_embassaments_diaris_Baells-Llosa-Sant_Pon.csv
echo.
echo Done. The manifests are in %S%\manifest_*.csv
pause

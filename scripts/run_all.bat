@echo off
rem Windows version of scripts/run_all.sh (see the comments there).
chcp 65001 >nul
cd /d "%~dp0.."
set V=long
set RUN=python -m uv run --frozen --all-extras llobregat

%RUN% aca-merge || goto :error
%RUN% aca-rain --basins Llobregat || goto :error
%RUN% qc-apply || goto :error
%RUN% splits || goto :error
%RUN% build-dataset --variant %V% || goto :error
%RUN% analyse-features --variant %V% || goto :error
%RUN% train --variant %V% --models persistence climatology ridge || goto :error
%RUN% tune --variant %V% || goto :error
%RUN% tune --variant %V% --final || goto :error
%RUN% train --variant %V% --models lgbm --params data\processed\sant_joan_despi\%V%\tuning\final || goto :error
%RUN% combine --variant %V% --models lgbm_tuned ridge --name ensemble || goto :error
%RUN% compare --variant %V% || goto :error
%RUN% intervals --variant %V% || goto :error
%RUN% final-evaluation --variant %V%
%RUN% report --variant %V% || goto :error
%RUN% operational-model --variant %V% || goto :error
echo Done.
goto :eof
:error
echo Failed: see the message above.
exit /b 1

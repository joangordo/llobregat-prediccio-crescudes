# Com reproduir els resultats

## 1. Entorn

```bash
uv sync --frozen --all-extras     # versions exactes de uv.lock
uv run pytest
```

A Windows, si `uv` no és al PATH, feu servir `python -m uv`.

## 2. Dades

Les dades no es distribueixen amb el codi. Han d'anar a `data/raw/`:

| Carpeta | Contingut | Com obtenir-les |
|---|---|---|
| `data/raw/aca_opendata/` | Cabals de 5 min des del 1996, cabals diaris des del 1951, embassaments diaris des del 2000 | `llobregat aca-opendata ...`; `scripts/download_aca_history.bat` ho descarrega tot (es pot reprendre) |
| `data/raw/aca_rain/` | Històric dels pluviòmetres de l'ACA, un Excel per conca (`Precipitació_Llobregat.xlsx`) | L'ACA el lliura a petició; no és públic |
| `data/raw/xema/` | Lectures semihoràries de la XEMA des del 2009 | `llobregat xema ...`; només per a la variant `full` |

## 3. Procés

`scripts/run_all.sh` (o `scripts\run_all.bat`) executa tots els passos en ordre:
`aca-merge`, `aca-rain`, `qc-apply`, `splits`, `build-dataset`, `analyse-features`, `train`,
`tune`, `combine`, `compare`, `intervals`, `final-evaluation`, `report` i
`operational-model`. Les sortides van a `data/interim/` i a
`data/processed/sant_joan_despi/<variant>/`.

L'ordre `final-evaluation` no es deixa executar una segona vegada: els anys 2025–2026 ja
s'han fet servir, i qualsevol canvi s'ha d'avaluar amb anys nous (2027 en endavant).

## 4. Sortides de versions anteriors

Fins al novembre del 2026 el codi escrivia els noms de carpetes, fitxers i columnes en
català (`llarg/`, `incertesa/`, `prediccions.parquet`, `plec`, `horitzo_h`...). Per fer
servir aquestes sortides amb el codi actual sense tornar a executar res:

```bash
uv run python scripts/migrate_outputs_to_english.py          # prova: llista els canvis
uv run python scripts/migrate_outputs_to_english.py --apply
```

## 5. El model operatiu

`llobregat operational-model` reentrena el sistema amb tots els anys hidrològics complets
menys l'últim, que es reserva per calibrar els intervals, i desa a `operational/` els models
(LightGBM en text i Ridge en joblib), el `manifest.json` (variables, ajustos del CQR, llindar,
anys i versions de les llibreries) i les importàncies. L'aplicatiu
[llobregat-flood-forecast-live](https://github.com/joangordo/llobregat-flood-forecast-live) carrega aquests fitxers
amb `llobregat.operational.OperationalModel`.

# Millores respecte al TFG

El TFG (Grau d'Estadística, Universitat de Barcelona, 2026) va construir un sistema de
predicció del cabal del Llobregat a Sant Joan Despí d'1 a 6 hores vista, en R, amb un
LightGBM amb pèrdua Tweedie (model M7) i un aplicatiu operatiu en Shiny. Aquest projecte
manté la pregunta i l'estació, i ho refà amb més dades, més control i una validació més
estricta.

## Resum

| Aspecte | TFG | Aquest projecte |
|---|---|---|
| Llenguatge | Scripts d'R | Paquet de Python amb línia d'ordres, més de 100 tests i integració contínua |
| Dades de cabal | Les que donaven les APIs (registre curt) | Arxiu històric de l'ACA de 5 min des del 1996, rebut a l'octubre del 2026 |
| Pluja | Estacions del Meteocat (API); ERA5-Land avaluat i descartat | 9 pluviòmetres de l'ACA amb sèrie des del 1996; la XEMA, avaluada en una variant a part |
| Període d'entrenament | Pocs anys | 2000–2026 (26 anys hidrològics), amb el Glòria |
| Control de qualitat | Neteja de les dades de les APIs | Regles de 5 min revisades amb casos reals (pics, caigudes, excursions, unitats en L/s, sensors encallats i saturats, balanç diari) |
| Resolució i horitzons | Horària, +1 a +6 h | 30 min, +1, +3, +6, +12 i +24 h |
| Selecció d'entrades | Segons la disponibilitat a les APIs | Regla de cobertura (≥ 85 % del període) |
| Selecció de variables | Eliminació enrere per horitzó | Duplicats amb el període de desenvolupament i selecció de fonts per permutació de grups dins de cada plec |
| Objectiu | Nivell del cabal (Tweedie, p = 1,5) | Canvi relatiu (LightGBM) i canvi absolut (Ridge) |
| Model final | M7, LightGBM Tweedie (el M5, RF híbrid, era millor a +1 h) | Conjunt LightGBM + Ridge a parts iguals, el mateix a tots els horitzons |
| Pesos per a les crescudes | Esglaonats per quantils | Provats (inclosos els del TFG): no milloren; pesos uniformes |
| Hiperparàmetres | Optimització per coordenades | Optuna a la validació interna del primer plec; només s'adopten si milloren > 5 % |
| Validació | 5 plecs creixents per any hidrològic | El mateix principi, amb separacions de 7 dies, validació interna, any de calibració i dos anys reservats avaluats una sola vegada (amb bloqueig) |
| Comparació de models | Mètriques per plec | Mètriques de tots els anys de test junts, bootstrap per blocs mensuals i test de Diebold–Mariano |
| Incertesa | No (els intervals es van retirar) | Quantils amb LightGBM + CQR en escala logarítmica; intervals del 80 i 90 % |
| Avisos | No formalitzats | Dos nivells (prealerta i alerta), avaluats amb POD, FAR i CSI per episodis |
| Operatiu | Shiny en R, en un ordinador del departament | GitHub Actions i GitHub Pages, al núvol, cada 30 min, només amb dades de l'ACA (sense clau del Meteocat) |

## Resultats al mateix període de test

El TFG es va avaluar de juny del 2025 a maig del 2026. Aquest projecte el va reservar dins
del seu conjunt de test final i el va avaluar una sola vegada, amb el sistema ja tancat:

| NSE | +1 h | +3 h | +6 h |
|---|---|---|---|
| TFG (M7) | 0,85 | — | 0,62 |
| Persistència | 0,945 | 0,659 | 0,333 |
| **Aquest projecte** | **0,978** | **0,852** | **0,666** |
| Índex de persistència d'aquest projecte | 0,60 | 0,57 | 0,50 |

Als dos anys reservats sencers (octubre 2024 – setembre 2026) el NSE és 0,966 / 0,803 / 0,585
a +1 / +3 / +6 h, davant de 0,936 / 0,608 / 0,219 de la persistència.

Les dues avaluacions no són idèntiques (el TFG predeia a resolució horària i aquest projecte
a 30 min), però el període és el mateix. La diferència més important no és el NSE a +6 h,
sinó que el model del TFG no superava la persistència fins a +3 h, i aquest la supera a tots
els horitzons, també a +1 h.

## Què ha canviat i per què

**1. Més història.** Amb pocs anys, el model només havia vist unes poques crescudes grans.
L'arxiu de l'ACA des del 1996 dona 25 anys d'entrenament, amb tots els episodis importants.
Per aprofitar-lo cal que cada entrada tingui sèrie llarga: per això les entrades es trien per
cobertura i la XEMA (que comença el 2009) queda fora de la variant final.

**2. Control de qualitat.** L'arxiu històric té errors que no es veuen en un registre curt:
trams en L/s desats com a m³/s, sensors encallats durant setmanes, pics d'una sola lectura.
Sense corregir-los, un sol error de x1000 domina qualsevol mètrica quadràtica.

**3. Predir el canvi, no el nivell.** El TFG predeia directament el cabal amb una pèrdua
Tweedie. Els arbres no poden predir valors més alts que el màxim de l'entrenament, i a
horitzons curts el cabal futur és quasi el cabal actual. Predir el canvi relatiu respecte
al cabal actual soluciona les dues coses: amb les mateixes dades, el Tweedie dona un NSE de
0,59 a 1 h i el canvi relatiu, 0,99.

**4. Un model lineal al costat.** El Ridge sobre el canvi absolut extrapola millor els
pics; el LightGBM captura les no-linealitats. La mitjana a parts iguals, fixada abans de
mirar el test, és millor que tots dos per separat a la validació creuada.

**5. Validació més estricta.** Cada decisió (variables, fonts, hiperparàmetres, pesos,
mètode d'intervals, regla d'avís) es pren amb anys anteriors als de test, i els dos últims
anys es van avaluar una sola vegada. L'ordre que ho fa es bloqueja després de la primera
execució.

**6. Incertesa i avisos.** El TFG donava un sol valor. Ara cada predicció porta intervals
calibrats, i l'avís es basa en aquests intervals: a +3 h la prealerta detecta el 88 % dels
episodis, mentre que la predicció central només en detecta el 50 %.

**7. Coses que s'han provat i no funcionen.** Els pesos del TFG, la pèrdua Tweedie, els
arbres lineals, la variant amb totes les estacions des del 2018 i l'ACI s'han comparat amb
les mateixes files. Es documenten al README perquè també són resultats.

## Què queda pendent

Als anys reservats, el Ridge sol (sense el LightGBM) va tenir millor NSE que el conjunt d'1 a
12 h i errors de pic més petits. Com que el conjunt es va decidir abans de mirar aquests
anys, es manté; però és el primer canvi que caldrà provar amb dades del 2027 en endavant.
L'aplicatiu arxiva totes les prediccions precisament per poder fer aquesta avaluació.

## Què no ha canviat

La limitació física és la mateixa: les crescudes generades per pluja a la part baixa de la
conca arriben a Sant Joan Despí en 1,5–4 hores, i sense una predicció meteorològica no es
poden anticipar més enllà d'aquest temps. El TFG ja ho apuntava; aquí es quantifica amb el
retard i l'error dels pics per horitzó.

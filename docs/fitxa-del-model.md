# Fitxa del model: predicció de crescudes del Llobregat (v1.0)

## El model

- **Què prediu:** el cabal mitjà de 30 minuts (m³/s) del Llobregat a Sant Joan Despí
  (estació LB-60 de l'ACA, 4.908 km²) a 1, 3, 6, 12 i 24 hores vista, cada 30 minuts.
- **Predicció central:** mitjana a parts iguals d'un LightGBM del canvi relatiu del cabal i
  d'un Ridge del canvi absolut (una parella de models per horitzó).
- **Incertesa:** models quantílics de LightGBM (5, 10, 90 i 95 %) calibrats amb CQR asimètric
  en escala logarítmica: intervals del 80 % i del 90 %.
- **Avisos:** llindar = percentil 99 dels cabals d'entrenament: 87 m³/s per al model avaluat
  i 86 m³/s per al model operatiu (un cabal alt, no necessàriament una crescuda). Prealerta: el límit superior de
  l'interval del 90 % supera el llindar. Alerta: la predicció central el supera.
- **Autor:** Joan Gordó Guardia (Universitat de Barcelona). **Versió:** 1.0, novembre 2026.

## Ús previst

- Recerca, docència i demostració de la predicció de crescudes amb dades.
- Una visió experimental i complementària de com pot evolucionar el riu les hores següents.

**Fora d'abast:** no és un sistema oficial d'avís i no s'ha de fer servir per prendre
decisions de seguretat. No s'ha provat en altres estacions ni rius.

## Dades d'entrenament

- Dades obertes de l'ACA de 5 min des del 1996: cabal a l'estació objectiu i a sis estacions
  de dalt triades per cobertura (Castellbell, Balsareny, Guardiola, l'Anoia a Sant Sadurní,
  Mediona i el Cardener a Súria); sortida i volum diaris de la Baells, la Llosa del Cavall
  i Sant Ponç (del dia anterior complet).
- Pluviòmetres de l'ACA de 5 min des del 1996 (9 a la conca, lliurats per l'ACA).
- Període d'entrenament del model avaluat: anys hidrològics 2001–2023, i el 2024 per
  calibrar els intervals. El model operatiu es reentrena amb 2001–2025 i es calibra amb el
  2026.

## Avaluació

| NSE | +1 h | +3 h | +6 h | +12 h | +24 h |
|---|---|---|---|---|---|
| Validació creuada 2020–2024 | 0,992 | 0,938 | 0,825 | 0,667 | 0,427 |
| Validació creuada, persistència | 0,983 | 0,899 | 0,765 | 0,549 | 0,193 |
| Anys reservats 2025–2026 | 0,966 | 0,803 | 0,585 | 0,441 | 0,272 |
| Anys reservats, persistència | 0,936 | 0,608 | 0,219 | −0,026 | −0,244 |
| Anys reservats, només Ridge (referència) | 0,971 | 0,815 | 0,632 | 0,451 | 0,217 |

Pics dels 32 episodis dels anys reservats (cabal per sobre de 87 m³/s):

| | +1 h | +3 h | +6 h | +12 h | +24 h |
|---|---|---|---|---|---|
| Error del pic (mediana) | +0,3 % | −14,5 % | −27,6 % | −35,7 % | −46,6 % |
| Retard del pic (mediana) | 0,5 h | 2,5 h | 6 h | 12 h | 23,5 h |

Avisos als anys reservats (32 episodis):

| | Emès abans de superar el llindar | Antelació (mediana) |
|---|---|---|
| Prealerta (límit superior de l'interval del 90 %) | 29 / 32 | 3,5 h |
| Alerta (predicció central) | 21 / 32 | 2 h |

| POD / FAR per horitzó | +1 h | +3 h | +6 h |
|---|---|---|---|
| Alerta | 0,84 / 0,07 | 0,50 / 0,38 | 0,38 / 0,43 |
| Prealerta | 1,00 / 0,36 | 0,88 / 0,55 | 0,56 / 0,80 |

Al període de test del TFG (juny 2025 – maig 2026), dins dels anys reservats, el NSE és
0,978 / 0,852 / 0,666 a +1 / +3 / +6 h (persistència: 0,945 / 0,659 / 0,333).

Cobertura dels intervals als anys reservats: 0,86–0,90 (80 %) i 0,94–0,96 (90 %); amb cabals
alts (per sobre del percentil 95), la de l'interval del 80 % baixa de 0,82 a +1 h a 0,52 a
+24 h.

## Limitacions

- **Pics tardans i curts a horitzons llargs:** als anys reservats, des de +6 h el pic previst
  arriba gairebé tan tard com el de la persistència i un 28 % o més curt. El model només veu
  la pluja que ja ha caigut.
- **Extrapolació:** la crescuda més gran registrada (Glòria, gener 2020, uns 1.470 m³/s en
  mitjana de 30 min) marca el límit del que han vist els arbres.
- **Any de calibració:** els intervals del model avaluat es van calibrar amb el 2024, un any
  de sequera, i van sortir una mica amples.
- **Dades en temps real:** si falta una estació de dalt, la predicció empitjora; l'aplicatiu
  mostra l'antiguitat de les dades i no fa predicció si falta l'estació objectiu.
- **Canvis a la conca:** l'operació dels embassaments o noves captacions poden canviar la
  relació entre entrades i cabal. Els anys 2025–2026 ja s'han fet servir; qualsevol canvi
  s'ha d'avaluar amb anys nous.

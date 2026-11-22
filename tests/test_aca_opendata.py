from datetime import date

import pandas as pd
import pytest

from llobregat.ingest import aca_opendata as ao
from llobregat.ingest.xema_opendata import SocrataClient

# Schema of the logging-interval datasets (3yr3-vq6y, 9evs-b5nk, mmzz-keb8)
ROWS = [
    {
        "dia": "2024-01-01T00:00:00.000",
        "hora": "00:05",
        "codi_estacio": "EA049_C4049",
        "estacio": "Aforament - Sant Joan Despí",
        "conca": "El Llobregat",
        "subconca": "Baix Llobregat",
        "tipus_variable": "Cabal riu",
        "codi_variable": "4652845",
        "descripcio_variable": "EA049_Sant Joan Despí_Cabal riu",
        "unitat_mesura": "m³/s",
        "nivell_qualitat": "NQ_REV0",
        "valor": "12.5",
        "utm_x": "421000",
    },
    {
        "dia": "2024-01-01T00:00:00.000",
        "hora": "00:10",
        "codi_estacio": "EA023",
        "estacio": "Aforament - Castellbell i el Vilar",
        "conca": "El Llobregat",
        "subconca": "Baix Llobregat",
        "tipus_variable": "Nivell riu",
        "codi_variable": "4652831",
        "descripcio_variable": "EA023_Castellbell_Nivell riu",
        "unitat_mesura": "m",
        "nivell_qualitat": "NQ_REV0",
        "valor": "0.8",
        "utm_x": "400000",
    },
]

# Schema of the daily dataset 2bwe-vgpr
DAILY = [
    {
        "dia": "1990-03-01T00:00:00.000",
        "estaci": "Embassament de la Baells (Cercs)",
        "conca": "EL LLOBREGAT",
        "subconca": "ALT LLOBREGAT",
        "codi_variable": "4140501",
        "descripci_variable": "E14_La Baells_Cabal sortida",
        "cabal_riu_m3_s": "3.2",
        "utm_x": "405000",
        "lat": "42.1",
        "long": "1.9",
    },
    {
        "dia": "1990-03-02T00:00:00.000",
        "estaci": "Embassament de la Baells (Cercs)",
        "conca": "EL LLOBREGAT",
        "subconca": "ALT LLOBREGAT",
        "codi_variable": "4140501",
        "descripci_variable": "E14_La Baells_Cabal sortida",
        "cabal_riu_m3_s": "3.4",
        "utm_x": "405000",
        "lat": "42.1",
        "long": "1.9",
    },
]


def test_schema_registre_and_daily():
    sc = ao.schema_from_rows(ROWS)
    assert (sc.date, sc.hour, sc.series, sc.values, sc.quality) == (
        "dia",
        "hora",
        "codi_variable",
        ["valor"],
        "nivell_qualitat",
    )
    sd = ao.schema_from_rows(DAILY)
    assert (sd.date, sd.hour, sd.series, sd.values) == (
        "dia",
        None,
        "codi_variable",
        ["cabal_riu_m3_s"],
    )
    assert "estaci" in sd.meta and "descripci_variable" in sd.meta


def test_combine_time():
    ts = ao.combine_time(
        pd.Series(["2026-11-20T00:00:00.000", "2026-11-20T00:00:00.000"]),
        pd.Series(["01:50", "24:00"]),
    )
    assert ts.iloc[0] == pd.Timestamp("2026-11-20 01:50", tz="UTC")
    assert ts.iloc[1] == pd.Timestamp("2026-11-21 00:00", tz="UTC")
    tsd = ao.combine_time(pd.Series(["1990-03-01T00:00:00.000"]), None)
    assert tsd.iloc[0] == pd.Timestamp("1990-03-01", tz="UTC")


def test_month_ranges():
    r = ao.month_ranges(date(2024, 11, 15), date(2025, 2, 1))
    assert r == [
        (date(2024, 11, 15), date(2024, 12, 1)),
        (date(2024, 12, 1), date(2025, 1, 1)),
        (date(2025, 1, 1), date(2025, 2, 1)),
    ]


def test_sample_windows_closed_dataset():
    args = ao.build_parser().parse_args(["stations", "--dataset", "cabals_2010_2019"])
    w = ao.sample_windows(ao.DATASETS["cabals_2010_2019"], args)
    assert len(w) == 10 and w[0] == (date(2010, 1, 1), date(2010, 2, 1))
    assert ao.sample_windows(ao.DATASETS["cabals_diaris"], args) == [None]


def _fake(rows, log=None):
    def fake_get(self, dataset, params):
        if log is not None:
            log.append(params)
        if params.get("$offset", 0) > 0:
            return []
        if params.get("$limit") == 200 and "$where" not in params:
            return rows
        if "$group" in params:
            meta = [f.strip() for f in params["$group"].split(",")]
            return [
                {
                    **{k: r.get(k) for k in meta},
                    "n_rows": "10",
                    "first_date": r["dia"],
                    "last_date": r["dia"],
                }
                for r in rows
            ]
        w = params["$where"]
        return [r for r in rows if f"'{r['codi_variable']}'" in w and f"'{r['dia'][:7]}" in w]

    return fake_get


def test_stations_registre(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(SocrataClient, "get", _fake(ROWS, calls))
    ao.main(["--out", str(tmp_path), "stations", "--dataset", "cabals"])
    st = pd.read_csv(tmp_path / "stations_cabals.csv", encoding="utf-8-sig", dtype=str)
    assert st.set_index("codi_variable")["use"].to_dict() == {"4652845": "1", "4652831": "0"}
    w = next(c["$where"] for c in calls if "$group" in c)
    assert "LLOBREGAT" in w and "dia >=" in w


def test_stations_name_filter_drops_default_conca(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(SocrataClient, "get", _fake(DAILY, calls))
    ao.main(["--out", str(tmp_path), "stations", "--dataset", "cabals_diaris", "--name", "Baells"])
    w = next(c["$where"] for c in calls if "$group" in c)
    assert w == "(upper(estaci) like '%BAELLS%')"
    st = pd.read_csv(tmp_path / "stations_cabals_diaris_Baells.csv", encoding="utf-8-sig")
    assert st["use"].tolist() == [1]


def test_data_registre(tmp_path, monkeypatch):
    monkeypatch.setattr(SocrataClient, "get", _fake(ROWS))
    st = tmp_path / "st.csv"
    pd.DataFrame({"codi_variable": ["4652845", "4652831"], "use": [1, 0]}).to_csv(st, index=False)
    ao.main(
        [
            "--out",
            str(tmp_path),
            "data",
            "--dataset",
            "cabals",
            "--stations-file",
            str(st),
            "--start",
            "2024-01-01",
            "--end",
            "2025-01-01",
        ]
    )
    base = tmp_path / "readings" / "dataset=cabals"
    df = pd.read_parquet(base / "station=4652845" / "year=2024.parquet")
    assert len(df) == 1
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:05", tz="UTC")
    assert df["value"].dtype.kind == "f"
    assert not (base / "station=4652831").exists()


def test_data_daily(tmp_path, monkeypatch):
    def fake_get(self, dataset, params):
        if params.get("$offset", 0) > 0:
            return []
        if "$where" not in params:
            return DAILY
        return [r for r in DAILY if f"'{r['codi_variable']}'" in params["$where"]]

    monkeypatch.setattr(SocrataClient, "get", fake_get)
    st = tmp_path / "st.csv"
    pd.DataFrame({"codi_variable": ["4140501"], "use": [1]}).to_csv(st, index=False)
    ao.main(
        ["--out", str(tmp_path), "data", "--dataset", "cabals_diaris", "--stations-file", str(st)]
    )
    df = pd.read_parquet(
        tmp_path / "readings" / "dataset=cabals_diaris" / "station=4140501" / "all.parquet"
    )
    assert df["value"].tolist() == [3.2, 3.4]
    assert df["timestamp"].iloc[0] == pd.Timestamp("1990-03-01", tz="UTC")


def test_unknown_dataset():
    with pytest.raises(SystemExit):
        ao.resolve_spec("no_existeix")
    assert ao.resolve_spec("hprm-xuw7").id == "hprm-xuw7"


def test_search(tmp_path, monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {
                        "resource": {
                            "id": "9evs-b5nk",
                            "name": "Cabals dels rius 2010-2019",
                            "attribution": "Agència Catalana de l'Aigua",
                            "type": "dataset",
                            "data_updated_at": "2026-10-23T00:00:00",
                            "columns_field_name": ["dia"],
                        }
                    },
                    {
                        "resource": {
                            "id": "aaaa-bbbb",
                            "name": "Turisme",
                            "attribution": "Idescat",
                            "type": "dataset",
                        }
                    },
                ]
            }

    c = SocrataClient()
    monkeypatch.setattr(c.session, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(ao, "SocrataClient", lambda **k: c)
    ao.main(["--out", str(tmp_path), "search", "--q", "cabal"])
    df = pd.read_csv(tmp_path / "aca_catalog.csv", encoding="utf-8-sig")
    assert df["id"].tolist() == ["9evs-b5nk"]

import pandas as pd

from llobregat.ingest.aca_history import HistoryMapping, load_aca_history
from llobregat.ingest.aca_sentilo import (
    clean_json_string,
    flatten_catalog,
    observations_to_frame,
    parse_sentilo_time,
)
from llobregat.ingest.meteocat_api import readings_to_frame
from llobregat.ingest.storage import append_dedup, last_timestamp, read_store


def test_sentilo_time_formats():
    a = parse_sentilo_time({"timestamp": "15/11/2026T10:30:00"})
    b = parse_sentilo_time({"time": 1794738600000})
    assert a == pd.Timestamp("2026-11-15 10:30", tz="UTC")
    assert b.tz is not None


def test_clean_json():
    assert clean_json_string('{"u": "m\\b5"}') == '{"u": "mm"}'


def test_catalog_and_observations():
    cat = {
        "providers": [
            {
                "provider": "AFORAMENT-EST",
                "sensors": [
                    {
                        "sensor": "S1",
                        "description": "Cabal",
                        "unit": "m3/s",
                        "location": "41.35 2.05",
                        "componentType": "aforament",
                        "componentAdditionalInfo": {"Riu": "Llobregat"},
                    }
                ],
            }
        ]
    }
    df = flatten_catalog(cat)
    assert df.loc[0, "lat"] == 41.35 and df.loc[0, "sensor"] == "S1"
    obs = [{"value": "12.5", "timestamp": "15/11/2026T10:30:00"}]
    o = observations_to_frame(obs, "AFORAMENT-EST", "S1", "flow")
    assert o.loc[0, "value"] == 12.5 and o.loc[0, "station"] == "AFORAMENT-EST/S1"


def test_meteocat_parsing():
    payload = [
        {
            "codi": "MQ",
            "variables": [
                {
                    "codi": 35,
                    "lectures": [
                        {
                            "data": "2026-11-15T10:00Z",
                            "valor": 1.2,
                            "estat": "V",
                            "baseHoraria": "SH",
                        }
                    ],
                }
            ],
        },
        {"codi": "ZZ", "variables": []},
    ]
    df = readings_to_frame(payload, ["MQ"])
    assert len(df) == 1 and df.loc[0, "variable"] == "35" and df.loc[0, "flag"] == "V|SH"


def test_store_append_dedup(tmp_path):
    t = pd.date_range(
        "2026-11-30 23:00", periods=4, freq="30min", tz="UTC"
    )  # crosses a month boundary
    df = pd.DataFrame(
        {
            "source": "aca",
            "station": "P/S",
            "variable": "flow",
            "timestamp": t,
            "value": [1.0, 2, 3, 4],
            "flag": None,
        }
    )
    assert append_dedup(df, tmp_path) == 4
    df2 = df.copy()
    df2["value"] = [9.0, 9, 9, 9]
    assert append_dedup(df2, tmp_path) == 0  # mateixes claus: substitueix, no afegeix
    back = read_store(tmp_path, "aca")
    assert len(back) == 4 and (back["value"] == 9).all()
    assert last_timestamp(tmp_path, "aca", "P/S", "flow") == t[-1]


def test_aca_history_local_time(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("data;cabal\n2026-07-01 12:00;10,5\n2026-07-01 12:05;11,0\n", encoding="utf-8")
    m = HistoryMapping(
        timestamp_col="data",
        value_col="cabal",
        station="SJD",
        timezone="Europe/Madrid",
        sep=";",
        decimal=",",
    )
    df = load_aca_history(path, m)
    assert df.loc[0, "timestamp"] == pd.Timestamp("2026-07-01 10:00", tz="UTC")  # CEST = UTC+2
    assert df.loc[0, "value"] == 10.5


def test_xema_history_split_hourly(tmp_path):
    from llobregat.ingest.xema_history import load_xema_history

    d = tmp_path / "readings" / "variable=35" / "station=U4"
    d.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "codi_estacio": "U4",
            "codi_variable": "35",
            "data_lectura": pd.to_datetime(
                ["2010-05-01 10:00", "2016-05-01 10:00", "2016-05-01 10:30"]
            ).tz_localize("UTC"),
            "valor_lectura": [4.0, 1.0, 2.0],
            "codi_estat": "V",
            "codi_base": ["HO", "SH", "SH"],
        }
    )
    raw.to_parquet(d / "year=2010.parquet")
    split = load_xema_history(tmp_path, ["U4"], hourly_policy="split")
    assert len(split) == 4 and split["value"].sum() == 7.0  # the rain total is preserved
    assert (split[split["flag"].str.endswith("HO_SPLIT")]["value"] == 2.0).all()
    drop = load_xema_history(tmp_path, ["U4"], hourly_policy="drop")
    assert len(drop) == 2


def test_realtime_series_from_config():
    from llobregat.config import load_station_config
    from llobregat.ingest.realtime import aca_series

    cfg = load_station_config("configs/stations/sant_joan_despi.yaml")
    series = aca_series([cfg])
    keys = {(r.sensor, v) for r, v in series}
    assert ("CALC001261", "flow") in keys  # target
    assert ("CALC000697", "volume") in keys  # La Baells
    assert ("082172-002-ANA014", "rain_intensity") in keys
    assert len(keys) == len(series)


def test_sentilo_blocks(monkeypatch):
    from datetime import UTC, datetime

    from llobregat.ingest.aca_sentilo import SentiloClient

    calls = []

    def fake(self, provider, sensor, start=None, end=None, limit=None):
        calls.append((start, end))
        return [{"value": "1", "time": int(start.timestamp() * 1000)}]

    monkeypatch.setattr(SentiloClient, "observations", fake)
    t0 = datetime(2026, 11, 13, tzinfo=UTC)
    obs = SentiloClient().observations_range("P", "S", t0, t0.replace(day=14, hour=6))
    assert len(calls) == 3 and len(obs) == 3  # 30 h in 12 h blocks

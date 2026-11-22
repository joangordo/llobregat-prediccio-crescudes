import numpy as np
import pandas as pd
import pytest

from llobregat.ingest import aca_rain_history as arh

openpyxl = pytest.importorskip("openpyxl")


def _write_xlsx(path, rows_sheet1, rows_sheet2):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Informe 1"
    ws.append([None, None, None, None])
    ws.append([None, None, "EA001_Estació U", "EA002_Estació Dos"])
    ws.append([None, None, "Intensitat precipitació", "Intensitat precipitació"])
    ws.append([None, None, "mm/h", "mm/h"])
    for r in rows_sheet1:
        ws.append([None, *r])
    ws2 = wb.create_sheet("Informe 1(1)")
    for r in rows_sheet2:
        ws2.append([None, *r])
    wb.save(path)


def test_parse_station():
    assert arh.parse_station("EA015_Sant Celoni") == ("EA015", "Sant Celoni")
    assert arh.parse_station("EA089_Fogars Selva") == ("EA089", "Fogars Selva")


def test_read_workbook_concatenates_sheets(tmp_path):
    t0 = pd.Timestamp("2015-01-01 00:00")
    s1 = [[(t0 + k * arh.STEP).to_pydatetime(), 0.0, 1.2] for k in range(10)]
    s2 = [[(t0 + k * arh.STEP).to_pydatetime(), 2.4, None] for k in range(10, 15)]
    f = tmp_path / "Precipitació_Prova.xlsx"
    _write_xlsx(f, s1, s2)
    df = arh.read_workbook(f)
    assert list(df.columns) == ["t", "EA001_Estació U", "EA002_Estació Dos"]
    assert len(df) == 15
    assert df["t"].is_monotonic_increasing
    assert df["EA002_Estació Dos"].isna().sum() == 5


def _sparse_then_dense():
    # 2009: only hourly rows (zeros) and one event; 2010: full grid
    hourly = pd.date_range("2009-01-01", "2009-12-31 23:00", freq="h")
    event = pd.date_range("2009-06-01 10:00", periods=6, freq="5min")
    dense = pd.date_range("2010-01-01", "2010-12-31 23:55", freq="5min")
    t = hourly.union(event).union(dense)
    v = np.zeros(len(t))
    v[t.isin(event)] = 12.0  # 1 mm per interval
    df = pd.DataFrame({"t": t, "EA001_X": v})
    gap = (df["t"] >= "2010-05-01") & (df["t"] < "2010-05-02")
    df.loc[gap, "EA001_X"] = np.nan  # real gap in the regular period
    return df


def test_dense_since_detects_regular_era():
    df = _sparse_then_dense()
    assert arh.dense_since(df["t"]) == pd.Timestamp("2010-01-01")


def test_to_regular_fills_implicit_zeros_only_in_sparse_era():
    df = _sparse_then_dense()
    wide, cut = arh.to_regular(df)
    s = wide["EA001_X"]
    assert cut == pd.Timestamp("2010-01-01")
    # event-based period: full grid with implicit zeros
    assert s[:"2009-12-31"].notna().all()
    assert s[:"2009-12-31"].sum() == pytest.approx(6 * 12.0)
    # regular period: the gap stays missing
    assert s["2010-05-01":"2010-05-01 23:55"].isna().all()


def test_to_regular_does_not_invent_zeros_after_long_silence():
    t = pd.DatetimeIndex(["2005-03-01 00:00", "2005-03-10 00:00"])
    df = pd.DataFrame({"t": t, "EA001_X": [0.0, 0.0]})
    wide, _ = arh.to_regular(df)
    s = wide["EA001_X"]
    # implicit zeros only in the HEARTBEAT minutes after the first reading
    assert s.notna().sum() == (arh.HEARTBEAT_MIN // 5 + 1) + 1
    assert s["2005-03-02":"2005-03-09"].isna().all()


def test_to_long_units_and_time_convention():
    idx = pd.date_range("2015-01-01", periods=3, freq="5min", name="t")
    wide = pd.DataFrame({"EA001_X": [0.0, 1.2, np.nan]}, index=idx)
    long = arh.to_long(wide, "Prova", "EA001_X")
    assert len(long) == 2
    assert long["value"].tolist() == [0.0, 0.1]
    assert str(long["station"].iloc[0]) == "EA001"
    assert long["timestamp"].iloc[0] == pd.Timestamp("2015-01-01 00:05", tz="UTC")

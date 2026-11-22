import pandas as pd

from llobregat.ingest import aca_merge as am


def _write(raw, ds, code, year, ts, values, nq="NQ_REV0"):
    d = raw / "readings" / f"dataset={ds}" / f"station={code}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(ts, tz="UTC"),
            "codi_variable": code,
            "value": values,
            "nivell_qualitat": nq,
        }
    ).to_parquet(d / f"year={year}.parquet", index=False)


def test_merge_segments_newer_wins(tmp_path):
    _write(
        tmp_path,
        "cabals_2010_2019",
        "1",
        2019,
        ["2019-12-31 23:55", "2020-01-01 00:00"],
        [1.0, 2.0],
    )
    _write(tmp_path, "cabals", "1", 2020, ["2020-01-01 00:00", "2020-01-01 00:05"], [9.0, 3.0])
    df = am.merge_segments(tmp_path, "1")
    assert df["value"].tolist() == [1.0, 9.0, 3.0]
    assert df["source"].tolist() == ["cabals_2010_2019", "cabals", "cabals"]


def test_hydro_year_and_coverage():
    ts = pd.Series(pd.date_range("2019-10-01", "2019-10-01 23:55", freq="5min", tz="UTC"))
    assert am.hydro_year(pd.DatetimeIndex(["2019-09-30", "2019-10-01"])).tolist() == [2019, 2020]
    cov = am.coverage_by_hydro_year(ts, [2020], until=pd.Timestamp("2019-10-03", tz="UTC"))
    assert abs(cov[2020] - 0.5) < 1e-9  # 1 day of data out of 2 elapsed days


def test_coverage_15min_counts_as_full():
    ts = pd.Series(
        pd.date_range("2019-10-01", "2019-10-02", freq="15min", tz="UTC", inclusive="left")
    )
    cov = am.coverage_by_hydro_year(ts, [2020], until=pd.Timestamp("2019-10-02", tz="UTC"))
    assert cov[2020] == 1.0


def test_merge_daily_chains_codes(tmp_path):
    for code, days, vals in [
        ("A", ["2024-12-30", "2024-12-31"], [1.0, 2.0]),
        ("B", ["2024-12-31", "2025-01-01"], [5.0, 6.0]),
    ]:
        d = tmp_path / "readings" / "dataset=cabals_diaris" / f"station={code}"
        d.mkdir(parents=True)
        pd.DataFrame({"timestamp": pd.DatetimeIndex(days, tz="UTC"), "value": vals}).to_parquet(
            d / "all.parquet", index=False
        )
    df = am.merge_daily(tmp_path, "cabals_diaris", ["A", "B"])
    assert df["value"].tolist() == [1.0, 5.0, 6.0]


def test_main_end_to_end(tmp_path, capsys):
    from llobregat.config import load_station_config

    cfg = load_station_config("configs/stations/sant_joan_despi.yaml")
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write(
        raw,
        "cabals",
        cfg.target.opendata,
        2020,
        pd.date_range("2020-01-01", periods=10, freq="5min"),
        [1.0] * 10,
    )
    am.main(["--raw", str(raw), "--out", str(out)])
    assert (out / "aca_flow" / f"{cfg.target.opendata}.parquet").exists()
    cov = pd.read_csv(out / "aca_flow" / "coverage.csv", encoding="utf-8-sig")
    assert "sant_joan_despi (target) · flow" in cov["name"].tolist()
    assert "Coverage per hydrological year" in capsys.readouterr().out


def test_unit_factors_detects_litres():
    t = pd.date_range("2020-01-01", "2020-01-20", freq="5min", tz="UTC", inclusive="left")
    q = pd.Series(10.0, index=t)  # real flow of 10 m³/s...
    q["2020-01-12 06:00":"2020-01-12 09:00"] = 400.0  # ... with a short flood
    raw = q.copy()
    raw["2020-01-05":"2020-01-15"] *= 1000  # eleven days in L/s (including the flood day)
    df5 = pd.DataFrame({"timestamp": t, "value": raw.to_numpy()})
    days = pd.date_range("2020-01-01", "2020-01-19", freq="D", tz="UTC")
    daily = q.resample("D").mean().reindex(days).to_frame("value").reset_index(names="timestamp")
    daily = daily.drop(index=[7, 8])  # two days without a daily value, within the L/s stretch
    out, info = am.apply_units(df5, daily)
    assert (out["value"] - q.to_numpy()).abs().max() < 1e-9
    assert info["days_ls_%"] > 0 and info["days_no_ref_%"] > 0
    assert (out["value_raw"] == df5["value"]).all()


def test_compact_years():
    assert am.compact_years([2001, 2002, 2003, 2007, 2009, 2010]) == "2001-2003, 2007, 2009-2010"

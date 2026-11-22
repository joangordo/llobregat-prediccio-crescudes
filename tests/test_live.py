"""Live forecast: Sentilo readings -> 30-min table -> features -> operational model."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from llobregat import analysis, live, operational
from llobregat.config import SensorRef
from test_block3 import setup  # noqa: F401  (shared fixture)

NOW = datetime(2020, 3, 1, 12, 7, tzinfo=UTC)


class FakeClient:
    """Sentilo stand-in: serves a 5-min series per sensor between start and end."""

    def __init__(self, series: dict[str, pd.Series]):
        self.series = series
        self.calls = []

    def observations_range(self, provider, sensor, start, end):
        self.calls.append((sensor, start, end))
        s = self.series.get(sensor)
        if s is None:
            return []
        s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        return [{"time": int(t.timestamp() * 1000), "value": str(v)} for t, v in s.items()]


def _cfg(setup_cfg):
    rain = setup_cfg.rain.model_copy(
        update={
            "aca_gauges": ["E14"],
            "aca_realtime": {"E14": SensorRef(provider="PLUVIOMETREACA-EST", sensor="R1")},
        }
    )
    return setup_cfg.model_copy(update={"rain": rain})


def _fake_series(start, end):
    t = pd.date_range(start, end, freq="5min", tz="UTC")
    q = pd.Series(20.0 + np.sin(np.arange(len(t)) / 300), index=t)
    q.iloc[-200] = 5000.0  # an isolated spike the quality control must remove
    up = q.shift(48).bfill() * 0.8
    rain = pd.Series(0.0, index=t)
    rain.iloc[-30:-24] = 12.0  # 6 readings of 12 mm/h = 6 mm in 30 min
    out = pd.Series(3.0, index=t)
    return {"CALC001261": q, "CALC001244": up, "R1": rain, "CALC000736": out}


INPUTS = {"flows": ["castellbell"], "reservoir_outflow": ["la_baells"], "rain_aca": ["E14"]}


def test_live_table_reproduces_the_training_conventions(setup, tmp_path):  # noqa: F811
    cfg = _cfg(setup[0])
    series = live.live_series(cfg, INPUTS)
    assert [s.key for s in series] == ["q", "up:castellbell", "res:la_baells:out", "rain:E14"]
    client = FakeClient(_fake_series("2020-01-01", "2020-03-01 12:05"))
    raw = live.update_cache(series, tmp_path, client, now=NOW, days=10)
    assert raw["q"]["timestamp"].min() >= pd.Timestamp("2020-02-20 12:07", tz="UTC")
    table = live.live_table(cfg, series, raw, now=NOW, days=10)
    assert table.index[-1] == pd.Timestamp("2020-03-01 12:00", tz="UTC")
    assert table["q"].max() < 100  # the 5000 m³/s spike was removed
    assert table["rain:E14"].sum() == pytest.approx(6.0)
    assert table["res:la_baells:out"].dropna().eq(3.0).all()
    # Incremental: a second run only asks for the readings after the cached ones
    client.calls.clear()
    live.update_cache(series, tmp_path, client, now=NOW, days=10)
    assert all(start > datetime(2020, 3, 1, tzinfo=UTC) for _, start, _ in client.calls)


def test_live_forecast_with_the_operational_model(setup, tmp_path):  # noqa: F811
    cfg, variant, root, _ = setup
    cfg = _cfg(cfg)
    analysis.run(cfg, variant, root, sample=40_000)
    out = operational.train_operational(cfg, variant, root, [1], max_rounds=60, rapid=True)
    man = json.loads((out / "manifest.json").read_text())
    assert "q" in man["input_ranges"] and man["versions"]["lightgbm"] != "unknown"
    man["inputs"] = INPUTS  # the synthetic dataset has no meta.json inputs
    (out / "manifest.json").write_text(json.dumps(man))
    model = operational.OperationalModel.load(out)
    series = live.live_series(cfg, INPUTS)
    client = FakeClient(_fake_series("2020-01-01", "2020-03-01 12:05"))
    raw = live.update_cache(series, tmp_path, client, now=NOW, days=30)
    fc, table = live.forecast(cfg, model, raw, series, now=NOW, history_h=6)
    assert len(fc) == 13 and fc["central"].notna().all()  # 6 h of issue times every 30 min
    assert (fc["lo90"] <= fc["hi90"]).all()
    status = live.input_status(series, raw, table, model.manifest["input_ranges"], now=NOW)
    assert status.set_index("key").loc["q", "check"] == "ok"
    assert status["age_min"].max() <= 10

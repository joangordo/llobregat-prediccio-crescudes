"""Real-time logging: download the new ACA and Meteocat data.

Meant to run every 30 minutes on GitHub Actions. Each run requests the observations after
the last stored one for each series and appends them to the store.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from llobregat.config import SensorRef, StationConfig
from llobregat.ingest.aca_sentilo import SentiloClient, observations_to_frame
from llobregat.ingest.meteocat_api import MeteocatClient, readings_to_frame
from llobregat.ingest.storage import append_dedup, last_timestamp

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK = timedelta(hours=24)  # if a series has no stored data yet
MAX_CATCHUP = timedelta(days=7)  # maximum catch-up after an outage
XEMA_PRECIP = 35


def _is_set(ref: SensorRef | None) -> bool:
    return ref is not None and "TODO" not in (ref.provider, ref.sensor)


def aca_series(configs: list[StationConfig]) -> list[tuple[SensorRef, str]]:
    """All the ACA series to log (no duplicates or TODO)."""
    series: dict[tuple[str, str, str], tuple[SensorRef, str]] = {}
    for cfg in configs:
        candidates = [(cfg.target, "flow")]
        candidates += [(u.aca, "flow") for u in cfg.upstream]
        for r in cfg.reservoirs:
            candidates += [(r.outflow, "outflow"), (r.volume, "volume"), (r.level, "level")]
        candidates += [(s, s.variable) for s in cfg.log_only]
        for ref, var in candidates:
            if _is_set(ref):
                plain = SensorRef(provider=ref.provider, sensor=ref.sensor)
                series[(ref.provider, ref.sensor, var)] = (plain, var)
    return list(series.values())


def fetch_aca(
    configs: list[StationConfig], store: Path, client: SentiloClient | None = None
) -> int:
    client = client or SentiloClient()
    now = datetime.now(UTC)
    frames = []
    for ref, var in aca_series(configs):
        station = f"{ref.provider}/{ref.sensor}"
        last = last_timestamp(store, "aca", station, var)
        start = (
            (last.to_pydatetime() + timedelta(seconds=1))
            if last is not None
            else now - DEFAULT_LOOKBACK
        )
        try:
            obs = client.observations_range(ref.provider, ref.sensor, start=start, end=now)
        except RuntimeError as exc:  # a failing series does not stop the others
            log.error("ACA %s (%s): %s", station, var, exc)
            continue
        df = observations_to_frame(obs, ref.provider, ref.sensor, var)
        log.info("ACA %s (%s): %d observations", station, var, len(df))
        frames.append(df)
    if not frames:
        return 0
    return append_dedup(pd.concat(frames, ignore_index=True), store)


def fetch_xema(
    configs: list[StationConfig], store: Path, client: MeteocatClient | None = None
) -> int:
    stations = sorted({s for cfg in configs for s in cfg.rain.xema_stations})
    if not stations:
        return 0
    client = client or MeteocatClient()
    now = datetime.now(UTC)
    days = [now.date()]
    if now.hour < 2:  # just after midnight, also collect yesterday's last readings
        days.insert(0, (now - timedelta(days=1)).date())
    frames = []
    for day in days:
        payload = client.variable_day(XEMA_PRECIP, day)
        frames.append(readings_to_frame(payload, stations))
    df = pd.concat(frames, ignore_index=True)
    log.info(
        "XEMA: %d readings from %d stations", len(df), df["station"].nunique() if len(df) else 0
    )
    return append_dedup(df, store)


def run_realtime(configs: list[StationConfig], store: str | Path) -> dict[str, int]:
    """Run one logging round. An error in one source does not stop the other."""
    store = Path(store)
    result = {}
    for name, fn in (("aca", fetch_aca), ("xema", fetch_xema)):
        try:
            result[name] = fn(configs, store)
        except Exception as exc:
            log.error("Source %s failed: %s", name, exc)
            result[name] = -1
    return result

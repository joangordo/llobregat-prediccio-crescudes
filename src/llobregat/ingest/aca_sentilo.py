"""Client for the ACA real-time API (Sentilo platform).

Base: https://aplicacions.aca.gencat.cat/sdim2/apirest
  - catalog?componentType=<type>    → catalogue of providers and sensors
  - data/<provider>/<sensor>        → observations of a sensor

ACA user manual: "us_serveis_dades_API_REST.pdf" (ACA real-time open data page).

Behaviour verified with real responses (13/11/2026):
  - Each request returns at most 200 observations, from the most recent to the oldest,
    even if a higher limit is requested. That is why data are downloaded in 12 h blocks
    (144 observations at 5 minutes).
  - 'time' (epoch in ms) and 'timestamp' (dd/MM/yyyyTHH:mm:ss) agree and are UTC.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://aplicacions.aca.gencat.cat/sdim2/apirest"
SENTILO_TS_FORMAT = "%d/%m/%YT%H:%M:%S"
MAX_OBS_PER_REQUEST = 200
BLOCK = timedelta(hours=12)


def clean_json_string(text: str) -> str:
    """Clean invalid escape sequences returned by the ACA API."""
    text = re.sub(r"\\b5", "m", text)
    text = re.sub(r"\\ba", "º", text)
    return re.sub(r'\\[^\\"bfnrtu/]', "", text)


def parse_sentilo_time(obs: dict) -> pd.Timestamp | None:
    """Convert the timestamp of an observation to UTC.

    Sentilo may give 'time' (epoch milliseconds) and/or 'timestamp' (dd/MM/yyyyTHH:mm:ss).
    UTC is assumed, which is what Sentilo uses internally.
    """
    if obs.get("time") not in (None, ""):
        return pd.Timestamp(int(obs["time"]), unit="ms", tz="UTC")
    ts = str(obs.get("timestamp", "")).strip()
    if not ts:
        return None
    ts = ts.replace("Z", "").replace("UTC", "").strip()
    for fmt in (SENTILO_TS_FORMAT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.Timestamp(datetime.strptime(ts, fmt), tz="UTC")
        except ValueError:
            continue
    log.warning("Unknown date format: %s", ts)
    return None


@dataclass
class SentiloClient:
    base_url: str = BASE_URL
    timeout: int = 60
    max_retries: int = 5
    pause: float = 0.2

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                wait = min(2**attempt, 60)
                log.warning("Network error (%s); retry %d in %ds", exc, attempt, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                time.sleep(self.pause)
                text = resp.content.decode("utf-8", errors="replace")
                return json.loads(clean_json_string(text)) if text.strip() else {}
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2**attempt, 60)
                log.warning("HTTP %s; retry %d in %ds", resp.status_code, attempt, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {resp.status_code} at {resp.url}: {resp.text[:300]}")
        raise RuntimeError(f"Retries exhausted for {url}")

    def catalog(self, component_type: str | None = None) -> dict:
        params = {"componentType": component_type} if component_type else None
        return self._get("catalog", params)

    def observations(
        self,
        provider: str,
        sensor: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        params: dict[str, str | int] = {}
        if start:
            params["from"] = start.astimezone(UTC).strftime(SENTILO_TS_FORMAT)
        if end:
            params["to"] = end.astimezone(UTC).strftime(SENTILO_TS_FORMAT)
        if limit:
            params["limit"] = limit
        data = self._get(f"data/{provider}/{sensor}", params or None)
        return data.get("observations", []) if isinstance(data, dict) else []

    def observations_range(
        self, provider: str, sensor: str, start: datetime, end: datetime, block: timedelta = BLOCK
    ) -> list[dict]:
        """All observations between start and end, in blocks that do not exceed the limit."""
        out: list[dict] = []
        t0 = start
        while t0 < end:
            t1 = min(t0 + block, end)
            chunk = self.observations(provider, sensor, start=t0, end=t1, limit=1000)
            if len(chunk) >= MAX_OBS_PER_REQUEST:
                log.warning(
                    "%s/%s: %d observations between %s and %s; may be truncated",
                    provider,
                    sensor,
                    len(chunk),
                    t0,
                    t1,
                )
            out.extend(chunk)
            t0 = t1
        return out


def flatten_catalog(catalog: dict) -> pd.DataFrame:
    """Convert the catalogue JSON into a table (one row per sensor)."""
    rows = []
    for prov in catalog.get("providers", []):
        provider = prov.get("provider")
        for s in prov.get("sensors", []):
            lat = lon = None
            loc = str(s.get("location") or "").replace(",", " ").split()
            if len(loc) >= 2:
                with contextlib.suppress(ValueError):
                    lat, lon = float(loc[0]), float(loc[1])
            rows.append(
                {
                    "provider": provider,
                    "sensor": s.get("sensor"),
                    "description": s.get("description"),
                    "type": s.get("type"),
                    "unit": s.get("unit"),
                    "data_type": s.get("dataType"),
                    "component": s.get("component"),
                    "component_type": s.get("componentType"),
                    "lat": lat,
                    "lon": lon,
                    "additional_info": json.dumps(
                        s.get("componentAdditionalInfo", {}), ensure_ascii=False
                    ),
                }
            )
    return pd.DataFrame(rows)


def observations_to_frame(
    observations: list[dict], provider: str, sensor: str, variable: str
) -> pd.DataFrame:
    """Convert Sentilo observations to the common long format."""
    rows = []
    for obs in observations:
        ts = parse_sentilo_time(obs)
        if ts is None:
            continue
        rows.append(
            {
                "source": "aca",
                "station": f"{provider}/{sensor}",
                "variable": variable,
                "timestamp": ts,
                "value": pd.to_numeric(obs.get("value"), errors="coerce"),
                "flag": None,
            }
        )
    return pd.DataFrame(
        rows, columns=["source", "station", "variable", "timestamp", "value", "flag"]
    )

"""Client for the Meteocat API, for (near) real-time XEMA data.

Requires a key (environment variable METEOCAT_API_KEY), which must never be in the code.
Endpoint: /xema/v1/variables/mesurades/{variable}/{any}/{mes}/{dia}
  → readings of one variable for every station over a day (a single request).
Documentation: https://apidocs.meteocat.gencat.cat/documentacio/dades-de-la-xema/
To be verified against a real response: exact JSON structure.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.meteo.cat/xema/v1"


@dataclass
class MeteocatClient:
    api_key: str = field(default_factory=lambda: os.environ.get("METEOCAT_API_KEY", ""))
    timeout: int = 60
    max_retries: int = 5

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError("Missing environment variable METEOCAT_API_KEY")
        self.session = requests.Session()
        self.session.headers["X-Api-Key"] = self.api_key

    def variable_day(self, variable: int | str, day: date) -> list[dict]:
        url = f"{BASE_URL}/variables/mesurades/{variable}/{day:%Y}/{day:%m}/{day:%d}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                wait = min(2**attempt, 60)
                log.warning("Network error (%s); retry %d in %ds", exc, attempt, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                raise RuntimeError("Meteocat API quota exhausted (HTTP 429)")
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(min(2**attempt, 60))
                continue
            raise RuntimeError(f"HTTP {resp.status_code} at {url}: {resp.text[:300]}")
        raise RuntimeError(f"Retries exhausted for {url}")


def readings_to_frame(payload: list[dict], stations: list[str] | None = None) -> pd.DataFrame:
    """Convert the API response to the common long format.

    Expected structure: [{"codi": "XX", "variables": [{"codi": 35, "lectures":
    [{"data": "...Z", "valor": 0.0, "estat": "V", "baseHoraria": "SH"}]}]}]
    """
    keep = set(stations) if stations else None
    rows = []
    for st in payload or []:
        code = st.get("codi")
        if keep is not None and code not in keep:
            continue
        for var in st.get("variables", []):
            for lec in var.get("lectures", []):
                rows.append(
                    {
                        "source": "xema",
                        "station": code,
                        "variable": str(var.get("codi")),
                        "timestamp": pd.to_datetime(lec.get("data"), utc=True),
                        "value": lec.get("valor"),
                        "flag": f"{lec.get('estat', '')}|{lec.get('baseHoraria', '')}",
                    }
                )
    return pd.DataFrame(
        rows, columns=["source", "station", "variable", "timestamp", "value", "flag"]
    )

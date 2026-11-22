"""Data ingestion: ACA (Sentilo and history) and Meteocat (XEMA)."""

# Common long format for every observation
OBS_COLUMNS = ["source", "station", "variable", "timestamp", "value", "flag"]

"""Compatibility wrapper: the code now lives in llobregat.ingest.xema_opendata.

uv run python scripts/download_xema.py <command> ...
uv run llobregat xema <command> ...          (equivalent)
"""

from llobregat.ingest.xema_opendata import main

if __name__ == "__main__":
    main()

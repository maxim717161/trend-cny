"""Small client for the MOEX ISS futures endpoints."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

ISS = "https://iss.moex.com/iss"
UA = "moex-futures-trend-research/1.0"


def get_json(url: str, retries: int = 4, timeout: int = 45) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.4 * (2**attempt))
    raise RuntimeError(f"ISS request failed: {url} ({last})")


def _rows(block: dict) -> list[dict]:
    cols = block["columns"]
    return [dict(zip(cols, row)) for row in block["data"]]


def fetch_specs() -> list[dict]:
    url = (
        f"{ISS}/engines/futures/markets/forts/securities.json"
        "?iss.meta=off&iss.only=securities"
        "&securities.columns=SECID,SHORTNAME,SECNAME,ASSETCODE,LOTVOLUME,MINSTEP,STEPPRICE,LASTTRADEDATE"
    )
    payload = get_json(url)
    return _rows(payload["securities"])


def fetch_history_day(day: str) -> list[dict]:
    """Daily FORTS history for every listed future. `day` is YYYY-MM-DD."""
    rows: list[dict] = []
    start = 0
    while True:
        url = (
            f"{ISS}/history/engines/futures/markets/forts/securities.json"
            f"?date={day}&iss.meta=off&start={start}"
        )
        payload = get_json(url)
        block = payload["history"]
        page = _rows(block)
        rows.extend(page)
        cursor = payload.get("history.cursor", {}).get("data")
        if not cursor:
            break
        _index, total, pagesize = cursor[0]
        if pagesize <= 0:
            break
        start += pagesize
        if start >= total:
            break
        if not page:
            break
    return rows


def fetch_candles(secid: str, start: str, end: str) -> list[dict]:
    """Hourly candles. ISS returns at most 500 rows, so follow the last begin."""
    rows: list[dict] = []
    seen: set[str] = set()
    cursor = start
    while True:
        query = urllib.parse.urlencode(
            {"interval": 60, "from": cursor, "till": end, "iss.meta": "off"},
            quote_via=urllib.parse.quote,
        )
        url = f"{ISS}/engines/futures/markets/forts/securities/{urllib.parse.quote(secid)}/candles.json?{query}"
        payload = get_json(url)
        page = _rows(payload["candles"])
        if not page:
            break
        fresh = 0
        for rec in page:
            begin = rec["begin"]
            if begin in seen:
                continue
            seen.add(begin)
            rows.append(rec)
            fresh += 1
        if len(page) < 500 or fresh == 0:
            break
        cursor = _plus_minute(page[-1]["begin"])
    return rows


def _plus_minute(stamp: str) -> str:
    # 'YYYY-MM-DD HH:MM:SS' -> same clock plus one minute, ISS-local.
    from datetime import datetime, timedelta

    parsed = datetime.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S")
    return (parsed + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")

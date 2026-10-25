"""Download the top-25 liquid MOEX futures and store hourly candles.

Liquidity is the sum of ruble turnover over the last 60 full sessions, across
every expiry of an asset code. Candles are hourly so the git history stays
small. The continuous file keeps every traded contract that showed up in the
sample; the roll is rebuilt by analyze, not baked in as a price jump.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

from research.futures_trend.iss import fetch_candles, fetch_history_day, fetch_specs
from research.futures_trend.names import display_name
from research.futures_trend.series import build_continuous, format_begin

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "futures-hourly"
CACHE = DATA / "cache"
HISTORY = CACHE / "history"
CANDLES = CACHE / "candles"

# Last completed week before this study (Sunday 25 Oct 2026, Europe/Kaliningrad).
SAMPLE_START = date(2024, 1, 1)
SAMPLE_END = date(2026, 10, 25)
RANK_FROM = date(2026, 7, 1)
TOP_N = 25
FULL_DAY_FRACTION = 0.5

_PRINT = threading.Lock()


def log(message: str) -> None:
    with _PRINT:
        print(message, flush=True)


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def wednesdays(start: date, end: date):
    day = start
    while day.weekday() != 2:
        day += timedelta(days=1)
    while day <= end:
        yield day
        day += timedelta(days=7)


def iso(day: date) -> str:
    return day.isoformat()


def load_history(day: date) -> list[dict]:
    path = HISTORY / f"{iso(day)}.json"
    if path.exists():
        return json.loads(path.read_text())
    compact = []
    for row in fetch_history_day(iso(day)):
        board = row.get("BOARDID")
        if board not in (None, "RFUD"):
            continue
        asset = row.get("ASSETCODE")
        if not asset:
            continue
        compact.append(
            {
                "secid": row.get("SECID") or "",
                "asset": asset,
                "shortname": row.get("SHORTNAME") or "",
                "volume": float(row.get("VOLUME") or 0),
                "value": float(row.get("VALUE") or 0),
                "oi": float(row.get("OPENPOSITION") or 0),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compact))
    return compact


def collect_days(days: list[date], workers: int) -> dict[date, list[dict]]:
    found: dict[date, list[dict]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(load_history, day): day for day in days}
        for future in as_completed(futures):
            day = futures[future]
            done += 1
            try:
                found[day] = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the rest of the calendar
                log(f"history {iso(day)} failed: {exc}")
                found[day] = []
            if done % 20 == 0 or done == len(days):
                log(f"history {done}/{len(days)}")
    return found


def ranking_days(history: dict[date, list[dict]]) -> list[date]:
    totals = {}
    for day, rows in history.items():
        if day < RANK_FROM:
            continue
        totals[day] = sum(row["value"] for row in rows)
    positive = [value for value in totals.values() if value > 0]
    if not positive:
        raise RuntimeError("MOEX history returned no turnover in the ranking window")
    positive.sort()
    median = positive[len(positive) // 2]
    full = [day for day in sorted(totals) if totals[day] > FULL_DAY_FRACTION * median]
    chosen = full[-60:]
    log(
        f"ranking sessions: {len(chosen)} "
        f"({iso(chosen[0])} .. {iso(chosen[-1])}), "
        f"median day {median / 1e9:.1f} млрд руб"
    )
    return chosen


def build_universe(history: dict[date, list[dict]], specs: list[dict], days: list[date]) -> None:
    value = defaultdict(float)
    volume = defaultdict(float)
    front_oi = defaultdict(float)
    example = {}
    for day in days:
        per_asset_oi = defaultdict(float)
        for row in history.get(day, []):
            asset = row["asset"]
            value[asset] += row["value"]
            volume[asset] += row["volume"]
            if row["oi"] > per_asset_oi[asset]:
                per_asset_oi[asset] = row["oi"]
            if row["value"] > example.get(asset, ({"value": -1},))[0].get("value", -1):
                example[asset] = (row, row["value"])
        for asset, oi in per_asset_oi.items():
            front_oi[asset] += oi

    spec_by_asset = defaultdict(list)
    for spec in specs:
        if spec.get("ASSETCODE"):
            spec_by_asset[spec["ASSETCODE"]].append(spec)

    ranked = sorted(value, key=value.get, reverse=True)[:TOP_N]
    DATA.mkdir(parents=True, exist_ok=True)
    universe_path = DATA / "universe.csv"
    with universe_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "asset",
                "name",
                "secname",
                "turnover_rub",
                "avg_daily_turnover_rub",
                "avg_front_oi",
                "volume_contracts",
                "minstep",
                "stepprice",
                "lotvolume",
                "example_secid",
            ],
        )
        writer.writeheader()
        for rank, asset in enumerate(ranked, start=1):
            family = spec_by_asset.get(asset, [])
            steps = [float(item["MINSTEP"]) for item in family if item.get("MINSTEP") not in (None, "")]
            prices = [float(item["STEPPRICE"]) for item in family if item.get("STEPPRICE") not in (None, "")]
            lots = [float(item["LOTVOLUME"]) for item in family if item.get("LOTVOLUME") not in (None, "")]
            secname = ""
            if family:
                secname = sorted(family, key=lambda item: item.get("LASTTRADEDATE") or "")[-1].get("SECNAME") or ""
            sample = example.get(asset, ({"secid": ""}, 0))[0]
            row = {
                "rank": rank,
                "asset": asset,
                "name": display_name(asset, secname),
                "secname": secname,
                "turnover_rub": f"{value[asset]:.0f}",
                "avg_daily_turnover_rub": f"{value[asset] / len(days):.0f}",
                "avg_front_oi": f"{front_oi[asset] / len(days):.0f}",
                "volume_contracts": f"{volume[asset]:.0f}",
                "minstep": f"{steps[0]:.8f}" if steps else "",
                "stepprice": f"{prices[0]:.8f}" if prices else "",
                "lotvolume": f"{lots[0]:.0f}" if lots else "",
                "example_secid": sample.get("secid", ""),
            }
            writer.writerow(row)
            log(
                f"{rank:2} {asset:12} {row['name']:28} "
                f"{value[asset] / len(days) / 1e9:8.2f} млрд/день"
            )

    # Contracts that actually traded on a sampled day, including the older Wednesdays.
    traded = defaultdict(lambda: {"value": 0.0, "shortname": ""})
    for rows in history.values():
        for row in rows:
            if row["asset"] not in value or row["asset"] not in set(ranked):
                continue
            if row["value"] <= 0 and row["volume"] <= 0:
                continue
            slot = traded[(row["asset"], row["secid"])]
            slot["value"] += row["value"]
            if row["shortname"]:
                slot["shortname"] = row["shortname"]

    contracts_path = DATA / "contracts.csv"
    with contracts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset", "secid", "shortname", "sampled_value_rub"])
        writer.writeheader()
        for (asset, secid), slot in sorted(traded.items(), key=lambda item: (item[0][0], -item[1]["value"])):
            writer.writerow(
                {
                    "asset": asset,
                    "secid": secid,
                    "shortname": slot["shortname"],
                    "sampled_value_rub": f"{slot['value']:.0f}",
                }
            )
    log(f"contracts to download: {len(traded)}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cached_candles(secid: str, start: str, end: str) -> list[dict]:
    path = CANDLES / f"{secid}.csv"
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    raw = fetch_candles(secid, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["begin", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for rec in raw:
            writer.writerow(
                {
                    "begin": rec["begin"],
                    "open": rec["open"],
                    "high": rec["high"],
                    "low": rec["low"],
                    "close": rec["close"],
                    "volume": rec["volume"],
                }
            )
    temporary.replace(path)
    return raw


def _as_float(value) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def stitch_asset(asset: str, secids: list[str], start: str, end: str) -> None:
    records = []
    for secid in secids:
        for rec in cached_candles(secid, start, end):
            records.append(
                {
                    "secid": secid,
                    "begin": rec["begin"],
                    "open": _as_float(rec["open"]),
                    "high": _as_float(rec["high"]),
                    "low": _as_float(rec["low"]),
                    "close": _as_float(rec["close"]),
                    "volume": _as_float(rec["volume"]),
                }
            )
    stitched = build_continuous(records)
    path = DATA / f"{asset}.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["begin", "secid", "open", "high", "low", "close", "volume", "ret", "roll"],
        )
        writer.writeheader()
        for bar in stitched:
            writer.writerow(
                {
                    "begin": format_begin(bar["begin"]),
                    "secid": bar["secid"],
                    "open": f"{bar['open']:.6f}",
                    "high": f"{bar['high']:.6f}",
                    "low": f"{bar['low']:.6f}",
                    "close": f"{bar['close']:.6f}",
                    "volume": f"{bar['volume']:.0f}",
                    "ret": "" if bar["ret"] is None else f"{bar['ret']:.8f}",
                    "roll": bar["roll"],
                }
            )
    returns = [bar["ret"] for bar in stitched if bar["ret"] is not None]
    if returns:
        mean = sum(returns) / len(returns)
        var = sum((item - mean) ** 2 for item in returns) / len(returns)
        std = var**0.5
    else:
        std = 0.0
    rolls = sum(bar["roll"] for bar in stitched)
    first = stitched[0]["begin"].date().isoformat() if stitched else "-"
    last = stitched[-1]["begin"].date().isoformat() if stitched else "-"
    log(
        f"stitched {asset:12} bars={len(stitched):6} rolls={rolls:3} "
        f"{first} .. {last} hourly_std={std:.5f} contracts={len(secids)}"
    )


def download(workers: int, only: set[str] | None) -> None:
    contracts = read_csv(DATA / "contracts.csv")
    by_asset = defaultdict(list)
    for row in contracts:
        if only and row["asset"] not in only:
            continue
        by_asset[row["asset"]].append(row["secid"])
    if not by_asset:
        raise RuntimeError("no contracts — run the universe stage first")
    start = SAMPLE_START.isoformat()
    end = (SAMPLE_END + timedelta(days=1)).isoformat()
    # Download contract by contract into the cache, then stitch. A second run
    # reuses whatever candles already landed.
    jobs = [(asset, secid) for asset, secids in by_asset.items() for secid in secids]
    log(f"candle requests: {len(jobs)} contracts, {start} .. {end}")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cached_candles, secid, start, end): (asset, secid) for asset, secid in jobs}
        for future in as_completed(futures):
            asset, secid = futures[future]
            done += 1
            try:
                rows = future.result()
            except Exception as exc:  # noqa: BLE001
                log(f"candles {secid} failed: {exc}")
                continue
            if done % 25 == 0 or done == len(jobs):
                log(f"candles {done}/{len(jobs)} last {secid} rows={len(rows)}")
    for asset, secids in sorted(by_asset.items()):
        stitch_asset(asset, secids, start, end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["universe", "candles", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--assets", nargs="*", default=None)
    args = parser.parse_args()
    only = set(args.assets) if args.assets else None

    if args.stage in ("universe", "all"):
        days = sorted(set(daterange(RANK_FROM, SAMPLE_END)) | set(wednesdays(SAMPLE_START, SAMPLE_END)))
        log(f"history days: {len(days)}")
        history = collect_days(days, args.workers)
        specs = fetch_specs()
        chosen = ranking_days(history)
        (DATA / "ranking_days.txt").write_text("\n".join(iso(day) for day in chosen) + "\n")
        build_universe(history, specs, chosen)

    if args.stage in ("candles", "all"):
        download(args.workers, only)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

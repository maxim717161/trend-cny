"""Front-month continuous futures from hourly contract bars."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta


def parse_begin(stamp: str) -> datetime:
    """ISS futures candles are stamped in Moscow exchange time, without a zone."""
    return datetime.strptime(stamp.replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S")


def format_begin(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S+03:00")


def drop_spikes(bars: list[dict], jump: float = 0.40, back: float = 0.05) -> list[dict]:
    """Drop a single bar that jumps and immediately retraces. Neighbours stay."""
    if len(bars) < 3:
        return list(bars)
    keep = [True] * len(bars)
    for i in range(1, len(bars) - 1):
        prev = bars[i - 1]["close"]
        cur = bars[i]["close"]
        nxt = bars[i + 1]["close"]
        if prev <= 0 or cur <= 0 or nxt <= 0:
            keep[i] = False
            continue
        if abs(cur / prev - 1.0) > jump and abs(nxt / prev - 1.0) < back:
            keep[i] = False
    return [bar for bar, ok in zip(bars, keep) if ok]


def build_continuous(
    records: list[dict],
    hysteresis: float = 1.15,
    max_gap: timedelta = timedelta(days=12),
) -> list[dict]:
    """Stitch the active contract into one hourly series.

    Active contract of a Moscow date is the one with the highest hourly volume,
    and it is replaced only when another contract trades at least `hysteresis`
    times more. Returns are close-to-close of the same contract, so a roll does
    not book the gap between two prices. The first return after a gap longer
    than `max_gap` is left empty.
    """
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        close = rec["close"]
        volume = rec["volume"]
        if close is None or close <= 0 or volume is None or volume <= 0:
            continue
        begin = rec["begin"]
        if isinstance(begin, str):
            begin = parse_begin(begin)
        by_sec[rec["secid"]].append(
            {
                "secid": rec["secid"],
                "begin": begin,
                "open": rec["open"],
                "high": rec["high"],
                "low": rec["low"],
                "close": float(close),
                "volume": float(volume),
            }
        )

    for secid, bars in by_sec.items():
        bars.sort(key=lambda bar: bar["begin"])
        deduped: list[dict] = []
        seen: set[datetime] = set()
        for bar in bars:
            if bar["begin"] in seen:
                continue
            seen.add(bar["begin"])
            deduped.append(bar)
        by_sec[secid] = drop_spikes(deduped)

    day_volume: dict = defaultdict(lambda: defaultdict(float))
    for secid, bars in by_sec.items():
        for bar in bars:
            day_volume[bar["begin"].date()][secid] += bar["volume"]

    active: dict = {}
    current: str | None = None
    for day in sorted(day_volume):
        volumes = day_volume[day]
        best = max(volumes, key=volumes.get)
        if current is None or current not in volumes:
            current = best
        elif best != current and volumes[best] > hysteresis * volumes.get(current, 0.0):
            current = best
        active[day] = current

    own_return: dict[tuple[str, datetime], float | None] = {}
    for secid, bars in by_sec.items():
        previous = None
        for bar in bars:
            ret = None
            if previous is not None and previous["close"] > 0:
                gap = bar["begin"] - previous["begin"]
                if timedelta(0) < gap <= max_gap:
                    ret = bar["close"] / previous["close"] - 1.0
            own_return[(secid, bar["begin"])] = ret
            previous = bar

    stitched: list[dict] = []
    previous_secid = None
    for day in sorted(active):
        secid = active[day]
        for bar in by_sec[secid]:
            if bar["begin"].date() != day:
                continue
            stitched.append(
                {
                    "begin": bar["begin"],
                    "secid": secid,
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar["volume"],
                    "ret": own_return[(secid, bar["begin"])],
                    "roll": int(previous_secid is not None and secid != previous_secid),
                }
            )
            previous_secid = secid
    return stitched

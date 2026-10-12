#!/usr/bin/env python3
"""Annualized return of the two follower accounts from broker cash ledgers and MOEX closes."""

from __future__ import annotations

import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
CNY_RE = re.compile(r"CNY-|CNYRUBF|\bCR[HMUZ]\d\b")
FLOW_OPS = {
    "Пополнение счета",
    "Перевод зачисление",
    "Перевод списание",
    "Вывод средств",
    "Вывод средств (ПП)",
    "Вывод со счета",
}
OP_HINTS = (
    "РЕПО",
    "Комиссия",
    "Покупка/продажа",
    "Вариационная маржа",
    "Перевод",
    "Пополнение",
    "Вывод",
    "Выплата",
    "DFP/RFP",
    "Налог",
    "Дивиденд",
    "Отражение",
    "Купон",
)


def sheet_rows(path: Path) -> list[list[str]]:
    rows = []
    with zipfile.ZipFile(path) as zf, zf.open("xl/worksheets/sheet1.xml") as fh:
        for _ev, el in ET.iterparse(fh, events=("end",)):
            if el.tag != NS + "row":
                continue
            texts = []
            for cell in list(el):
                if cell.tag != NS + "c":
                    continue
                parts = [t.text for t in cell.iter(NS + "t") if t.text]
                if not parts:
                    value = cell.find(NS + "v")
                    if value is not None and value.text:
                        parts.append(value.text)
                text = "".join(parts).replace("\n", " ").strip()
                if text:
                    texts.append(text)
            el.clear()
            if texts:
                rows.append(texts)
    return rows


def parse_cash_file(path: Path) -> tuple[float, list[dict]]:
    rows = sheet_rows(path)
    start = next(i for i, cells in enumerate(rows) if cells[0].startswith("2. Операции"))
    end = next(i for i, cells in enumerate(rows) if cells[0].startswith("3.1"))
    opening = 0.0
    for cells in rows[start : start + 8]:
        if cells and cells[0] == "RUB" and len(cells) >= 3 and NUM_RE.match(cells[1]):
            opening = float(cells[1])
            break
    events = []
    for cells in rows[start:end]:
        if not cells or not DATE_RE.match(cells[0]):
            continue
        op_at = None
        for index, token in enumerate(cells):
            if any(token.startswith(hint) for hint in OP_HINTS):
                op_at = index
                break
        if op_at is None:
            continue
        numbers = [float(token) for token in cells[op_at + 1 :] if NUM_RE.match(token)]
        if len(numbers) < 2:
            continue
        note = " ".join(token for token in cells[op_at + 1 :] if not NUM_RE.match(token))
        events.append(
            {
                "date": datetime.strptime(cells[0], "%d.%m.%Y").date(),
                "op": cells[op_at],
                "credit": numbers[0],
                "debit": numbers[1],
                "note": note,
            }
        )
    events.sort(key=lambda row: row["date"])
    return opening, events


def cash_curve(folder: Path) -> tuple[dict[date, float], dict[date, float], dict[date, float]]:
    """End-of-day cash, external flow (+ deposit), and CNY variation margin."""
    eod: dict[date, float] = {}
    flows: dict[date, float] = defaultdict(float)
    vm: dict[date, float] = defaultdict(float)
    cash = 0.0
    for path in sorted(folder.glob("*.xlsx")):
        opening, events = parse_cash_file(path)
        # Official opening is the previous report's carried balance.
        cash = opening
        by_day: dict[date, list[dict]] = defaultdict(list)
        for event in events:
            by_day[event["date"]].append(event)
        if not by_day:
            continue
        day = min(by_day)
        last = max(by_day)
        while day <= last:
            for event in by_day.get(day, []):
                cash += event["credit"] - event["debit"]
                if event["op"] in FLOW_OPS or event["op"].startswith("Перевод") or event["op"].startswith("Пополнение") or event["op"].startswith("Вывод"):
                    flows[day] += event["credit"] - event["debit"]
                if event["op"].startswith("Вариационная") and CNY_RE.search(event["note"]):
                    vm[day] += event["credit"] - event["debit"]
            eod[day] = cash
            day += timedelta(days=1)
    return eod, flows, vm


def load_ops(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    for row in rows:
        stamp = datetime.fromisoformat(row["d"].replace("Z", "+00:00")) + timedelta(hours=3)
        row["msk"] = stamp.replace(tzinfo=None)
        row["date"] = row["msk"].date()
    rows.sort(key=lambda row: row["msk"])
    return rows


def is_futures_trade(row: dict) -> bool:
    ticker = row.get("ticker") or ""
    return row["t"] in ("OPERATION_TYPE_BUY", "OPERATION_TYPE_SELL") and (
        row.get("kind") == "INSTRUMENT_TYPE_FUTURES" or ticker.startswith("CR") or ticker == "CNYRUBF"
    )


def security_qty(ops: list[dict]) -> dict[date, dict[str, float]]:
    pos: dict[str, float] = defaultdict(float)
    eod: dict[date, dict[str, float]] = {}
    last_px: dict[str, float] = {}
    current = None
    for row in ops:
        if row["date"] != current and current is not None:
            eod[current] = {k: v for k, v in pos.items() if abs(v) >= 0.5}
        current = row["date"]
        if row["t"] not in ("OPERATION_TYPE_BUY", "OPERATION_TYPE_SELL"):
            continue
        if is_futures_trade(row):
            continue
        qty = float(row["qty"] or 0)
        price = float(row["px"] or 0)
        if price and qty and abs(abs(row["pay"]) - price * qty) > max(2.0, 0.02 * price * qty):
            qty = abs(row["pay"]) / price
        ticker = (row.get("ticker") or "").replace("@", "")
        if not ticker:
            continue
        # T-Technologies split 10-for-1 in April 2026. The API has no corporate-action
        # row, so one pre-split share stays as one share and the book goes short by 9.
        if ticker == "T" and price < 500 and last_px.get("T", 0) > 1500 and pos["T"] > 0:
            pos["T"] *= 10
        if price:
            last_px[ticker] = price
        pos[ticker] += qty if row["t"].endswith("BUY") else -qty
    if current is not None:
        eod[current] = {k: v for k, v in pos.items() if abs(v) >= 0.5}
    return eod


def trade_marks(ops: list[dict]) -> dict[date, dict[str, float]]:
    """Last trade price of each cash security, for names with no MOEX close."""
    last: dict[str, float] = {}
    eod: dict[date, dict[str, float]] = {}
    current = None
    for row in ops:
        if row["date"] != current and current is not None:
            eod[current] = dict(last)
        current = row["date"]
        if row["t"] not in ("OPERATION_TYPE_BUY", "OPERATION_TYPE_SELL") or is_futures_trade(row):
            continue
        price = float(row["px"] or 0)
        ticker = (row.get("ticker") or "").replace("@", "")
        if ticker and price:
            last[ticker] = price
    if current is not None:
        eod[current] = dict(last)
    return eod


def extend_cash(eod: dict[date, float], flows: dict[date, float], vm: dict[date, float], ops: list[dict]) -> None:
    """Append cash after the last broker-report day. Skip the midnight VM already inside the report."""
    last = max(eod)
    cash = eod[last]
    day_cursor = last
    for row in ops:
        if row["date"] <= last:
            continue
        if is_futures_trade(row):
            continue
        # Clearing VM just after midnight belongs to the last report session.
        if row["date"] == last + timedelta(days=1) and "VARMARGIN" in row["t"] and row["msk"].hour < 9:
            continue
        if row["date"] != day_cursor:
            eod[day_cursor] = cash
            day_cursor = row["date"]
        cash += row["pay"]
        if row["t"] in {
            "OPERATION_TYPE_INPUT",
            "OPERATION_TYPE_OUTPUT",
            "OPERATION_TYPE_INP_MULTI",
            "OPERATION_TYPE_OUT_MULTI",
        }:
            flows[row["date"]] += row["pay"]
        if "VARMARGIN" in row["t"]:
            vm[row["date"]] += row["pay"]
    eod[day_cursor] = cash


def moex_closes(ticker: str, start: date, end: date) -> dict[date, float]:
    cache = Path("/tmp/px") / f"{ticker}.json"
    if cache.exists() and cache.stat().st_size > 2:
        raw = json.loads(cache.read_text())
        if raw.get("missing"):
            return {}
        if raw:
            return {date.fromisoformat(day): price for day, price in raw.items()}
    if ticker.startswith("SU"):
        boards = ("TQOB",)
    elif ticker in {"LQDT", "TMON", "TMOS", "TRUR", "TLCB"}:
        # LQDT moved from TQTF to TQBR in June 2026.
        boards = ("TQBR", "TQTF")
    else:
        boards = ("TQBR",)
    closes: dict[date, float] = {}
    for board in boards:
        closes.update(_moex_board(ticker, board, start, end))
    cache.parent.mkdir(parents=True, exist_ok=True)
    if closes:
        cache.write_text(json.dumps({day.isoformat(): price for day, price in closes.items()}))
    else:
        cache.write_text(json.dumps({"missing": True}))
    return closes


def _moex_board(ticker: str, board: str, start: date, end: date) -> dict[date, float]:
    import subprocess

    closes: dict[date, float] = {}
    offset = 0
    while True:
        url = (
            "https://iss.moex.com/iss/history/engines/stock/markets/shares/"
            f"boards/{board}/securities/{ticker}.json"
            f"?from={start.isoformat()}&till={end.isoformat()}&start={offset}&iss.meta=off"
        )
        try:
            raw = subprocess.check_output(["curl", "-sS", "--retry", "2", "--retry-delay", "1", "-m", "20", url], text=True)
            payload = json.loads(raw)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            break
        history = payload.get("history") or {}
        columns = history.get("columns") or []
        data = history.get("data") or []
        if not data or "TRADEDATE" not in columns:
            break
        date_at = columns.index("TRADEDATE")
        price_name = "LEGALCLOSEPRICE" if "LEGALCLOSEPRICE" in columns else "CLOSE"
        price_at = columns.index(price_name)
        # Bond quotes are a percent of the 1000 RUB face.
        scale = 10.0 if board == "TQOB" else 1.0
        for row in data:
            if row[price_at] is None:
                continue
            closes[date.fromisoformat(row[date_at])] = float(row[price_at]) * scale
        offset += len(data)
        if len(data) < 100:
            break
        time.sleep(0.05)
    return closes


def forward_prices(closes: dict[date, float], start: date, end: date) -> dict[date, float]:
    out = {}
    last = None
    # Seed with the last close on or before start.
    earlier = [day for day in closes if day <= start]
    if earlier:
        last = closes[max(earlier)]
    day = start
    while day <= end:
        if day in closes:
            last = closes[day]
        if last is not None:
            out[day] = last
        day += timedelta(days=1)
    return out


FLOW_TYPES = {
    "OPERATION_TYPE_INPUT",
    "OPERATION_TYPE_OUTPUT",
    "OPERATION_TYPE_INP_MULTI",
    "OPERATION_TYPE_OUT_MULTI",
}
FOLLOW_FEES = {"OPERATION_TYPE_TRACK_MFEE", "OPERATION_TYPE_TRACK_PFEE"}
BROKER_FEES = {"OPERATION_TYPE_BROKER_FEE"}


def api_ledger(ops: list[dict]) -> tuple[dict[date, float], dict[date, float], dict[date, float], dict[date, float], dict[date, float]]:
    """Trade-date cash. Futures buy/sell payments are notional and stay out of cash.

    The April 2022 withdrawal of 95 422 ₽ on the trend account is not in the broker
    ledger (the account opens at zero). Leaving it in shifts every later balance.
    """
    cash = 0.0
    eod: dict[date, float] = {}
    flows: dict[date, float] = defaultdict(float)
    vm: dict[date, float] = defaultdict(float)
    follow: dict[date, float] = defaultdict(float)
    broker_fee: dict[date, float] = defaultdict(float)
    current = None
    for row in ops:
        if row["date"] < date(2023, 1, 1) and row["t"] == "OPERATION_TYPE_OUTPUT" and row["pay"] < -10000:
            continue
        if row["date"] != current and current is not None:
            eod[current] = cash
        current = row["date"]
        if is_futures_trade(row):
            continue
        cash += row["pay"]
        if row["t"] in FLOW_TYPES:
            flows[row["date"]] += row["pay"]
        if "VARMARGIN" in row["t"]:
            vm[row["date"]] += row["pay"]
        if row["t"] in FOLLOW_FEES:
            follow[row["date"]] += row["pay"]
        if row["t"] in BROKER_FEES:
            broker_fee[row["date"]] += row["pay"]
    if current is not None:
        eod[current] = cash
    return eod, flows, vm, follow, broker_fee


def build_nav(folder: str, ops_path: Path) -> dict:
    _broker_cash, _broker_flows, broker_vm = cash_curve(ROOT / "broker-report" / folder)
    ops = load_ops(ops_path)
    traded, flows, api_vm, follow, broker_fee = api_ledger(ops)
    # Broker notes name the contract. The API posts that same variation margin
    # on the next morning, so the first API day after the report repeats the
    # last report day and must not be added again.
    vm = dict(broker_vm)
    report_end = max(broker_vm) if broker_vm else date(2026, 10, 7)
    for day, value in api_vm.items():
        if day > report_end + timedelta(days=1):
            vm[day] = value
    eod = {}
    cash = None
    day = min(traded)
    while day <= max(traded):
        if day in traded:
            cash = traded[day]
        if cash is not None and day >= date(2025, 1, 1):
            eod[day] = cash
        day += timedelta(days=1)
    qty_by_day = security_qty(ops)
    # Returns are measured from 2025, when the yuan books actually have capital.
    start = max(min(eod), date(2025, 1, 1))
    end = max(eod)
    # Carry security quantities across days without trades.
    known = [day for day in qty_by_day if day <= start]
    qty: dict[str, float] = dict(qty_by_day[max(known)]) if known else {}
    daily_qty: dict[date, dict[str, float]] = {}
    day = start
    while day <= end:
        if day in qty_by_day:
            qty = qty_by_day[day]
        daily_qty[day] = dict(qty)
        day += timedelta(days=1)
    tickers = sorted({ticker for book in daily_qty.values() for ticker, qty in book.items() if abs(qty) >= 0.5})
    prices = {}
    for ticker in tickers:
        closes = moex_closes(ticker, start, end)
        prices[ticker] = forward_prices(closes, start, end)
        print(f"  price {ticker} {len(closes)}", flush=True)
    mark_eod = trade_marks(ops)
    known_marks = [day for day in mark_eod if day <= start]
    mark: dict[str, float] = dict(mark_eod[max(known_marks)]) if known_marks else {}
    daily_mark: dict[date, dict[str, float]] = {}
    day = start
    while day <= end:
        if day in mark_eod:
            mark = mark_eod[day]
        daily_mark[day] = dict(mark)
        day += timedelta(days=1)
    nav = {}
    for day, cash in eod.items():
        value = cash
        for ticker, lots in daily_qty.get(day, {}).items():
            price = prices.get(ticker, {}).get(day)
            if price is None:
                price = daily_mark.get(day, {}).get(ticker)
            if price is None:
                continue
            value += lots * price
        nav[day] = value
    last_day = max(daily_qty)
    return {
        "nav": nav,
        "flows": flows,
        "vm": vm,
        "api_vm": api_vm,
        "follow": follow,
        "broker_fee": broker_fee,
        "prices": prices,
        "tickers": tickers,
        "last_qty": daily_qty[last_day],
    }


def annualize(factor: float, span: int) -> float | None:
    if span <= 0 or factor <= 0:
        return None
    return factor ** (365.25 / span) - 1


def window_stats(book: dict, start: date, end: date) -> dict:
    days = [day for day in sorted(book["nav"]) if start <= day <= end]
    if len(days) < 2:
        return {}
    # Variation margin uses the requested bounds. An empty evening before the
    # first deposit stays in this sum and drops out of the return chain below.
    vm = sum(value for day, value in book["vm"].items() if start < day <= end)
    api_vm = sum(value for day, value in book["api_vm"].items() if start < day <= end)
    follow = sum(value for day, value in book["follow"].items() if start < day <= end)
    broker_fee = sum(value for day, value in book["broker_fee"].items() if start < day <= end)
    funded = [day for day in days if book["nav"][day] >= 1000]
    if len(funded) < 2:
        return {}
    days = [day for day in days if day >= funded[0]]
    factor = 1.0
    prev = book["nav"][days[0]]
    worst = (0.0, days[0])
    jumps = []
    resets = 0
    for day in days[1:]:
        flow = book["flows"].get(day, 0.0)
        nav_now = book["nav"][day]
        # A withdrawn account is not a base. The refill day starts a new chain.
        if prev < 1000:
            resets += 1
            prev = nav_now
            continue
        growth = (nav_now - flow) / prev
        ret = growth - 1
        if abs(ret) > 0.08:
            jumps.append((day, ret, flow, prev, nav_now))
        if growth <= 0:
            factor = 0.0
            prev = nav_now
            continue
        factor *= growth
        if ret < worst[0]:
            worst = (ret, day)
        prev = nav_now
    span = (days[-1] - days[0]).days
    # The opening day's deposit is already inside the starting value.
    deposits = sum(book["flows"].get(day, 0.0) for day in days[1:])
    invested = [book["nav"][day] for day in days if book["nav"][day] >= 1000]
    avg_nav = sum(invested) / len(invested) if invested else 0.0
    lqdt = book["prices"].get("LQDT", {})
    lqdt_growth = None
    if lqdt.get(days[0]) and lqdt.get(days[-1]):
        lqdt_growth = lqdt[days[-1]] / lqdt[days[0]] - 1
    profit = book["nav"][days[-1]] - book["nav"][days[0]] - deposits
    return {
        "start": days[0],
        "end": days[-1],
        "span": span,
        "nav0": book["nav"][days[0]],
        "nav1": book["nav"][days[-1]],
        "deposits": deposits,
        "profit": profit,
        "twr": factor - 1,
        "cagr": annualize(factor, span),
        "vm": vm,
        "api_vm": api_vm,
        "other_vm": api_vm - vm,
        "follow": follow,
        "broker_fee": broker_fee,
        "avg_nav": avg_nav,
        "vm_on_nav": vm / avg_nav if avg_nav else 0.0,
        "vm_annual": (vm / avg_nav) * (365.25 / span) if avg_nav and span else 0.0,
        "follow_on_nav": follow / avg_nav if avg_nav else 0.0,
        "lqdt": lqdt_growth,
        "lqdt_cagr": annualize(1 + lqdt_growth, span) if lqdt_growth is not None else None,
        "jumps": jumps,
        "worst": worst,
        "resets": resets,
    }


def pct(value: float | None) -> str:
    if value is None or isinstance(value, complex) or value != value:
        return "—"
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return sign + f"{abs(value) * 100:.1f}%"


def money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") + " ₽"


def rub(value: float) -> str:
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return sign + f"{abs(value):,.0f}".replace(",", " ") + " ₽"


def render_window(title: str, stats: dict, note: str = "") -> list[str]:
    lines = [f"### {title}", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append(
        f"С {stats['start'].strftime('%d.%m.%Y')} по {stats['end'].strftime('%d.%m.%Y')}, {stats['span']} дней. "
        f"Стоимость счета {money(stats['nav0'])} → {money(stats['nav1'])}. "
        f"Вводы минус выводы {rub(stats['deposits'])}. "
        f"Прирост после вводов {rub(stats['profit'])}."
    )
    lines.append("")
    lines.append(f"- Доходность счета за период: **{pct(stats['twr'])}**.")
    if stats["span"] < 80:
        lines.append(
            f"- Если растянуть этот отрезок на год, темп вышел бы {pct(stats['cagr'])}. "
            f"Отрезок короткий, годовую ставку из него не выводят."
        )
    else:
        lines.append(f"- В годовых: **{pct(stats['cagr'])}**.")
        if stats["span"] < 300:
            lines.append("- Отрезок короче года: годовая цифра — тот же результат, растянутый сложным процентом.")
    if stats["lqdt_cagr"] is not None and stats["cagr"] is not None:
        lines.append(
            f"- LQDT за те же даты: {pct(stats['lqdt'])} за период, {pct(stats['lqdt_cagr'])} в годовых."
        )
        if stats["span"] < 80:
            lines.append(f"- Счет минус LQDT за период: **{pct(stats['twr'] - stats['lqdt'])}**.")
        else:
            lines.append(f"- Счет минус LQDT, в годовых: **{pct(stats['cagr'] - stats['lqdt_cagr'])}**.")
    vm_line = (
        f"- Вариационная маржа по фьючерсу на юань: {rub(stats['vm'])}. "
        f"К средней стоимости счета {money(stats['avg_nav'])} это {pct(stats['vm_on_nav'])} за период."
    )
    if stats["span"] >= 120:
        vm_line += f" В пересчете на год без сложного процента {pct(stats['vm_annual'])}."
    lines.append(vm_line)
    if abs(stats["follow"]) >= 1:
        lines.append(
            f"- Комиссия за следование: {rub(stats['follow'])}, {pct(stats['follow_on_nav'])} от средней стоимости. "
            f"Она уже вычтена из доходности счета и в маржу не входит."
        )
    if abs(stats["broker_fee"]) >= 100:
        lines.append(f"- Комиссия брокера за сделки: {rub(stats['broker_fee'])}. Она тоже уже вычтена из доходности счета.")
    if abs(stats["other_vm"]) >= 1500:
        lines.append(
            f"- Прочая вариационная маржа на счете за эти даты: {rub(stats['other_vm'])}. "
            f"Доходность счета включает и ее."
        )
    if stats["resets"]:
        lines.append(f"- Счет обнулялся и заводился заново, таких сбросов базы: {stats['resets']}.")
    if stats["jumps"]:
        lines.append("")
        lines.append("Дни, где стоимость сдвинулась больше чем на 8% после поправки на ввод:")
        for day, ret, flow, before, after in stats["jumps"][:8]:
            lines.append(
                f"- {day.strftime('%d.%m.%Y')}: {pct(ret)}, ввод {rub(flow)}, счет {money(before)} → {money(after)}."
            )
    lines.append("")
    return lines


def stitched_section(first: dict, second: dict) -> list[str]:
    if first["cagr"] is None or second["cagr"] is None or first["lqdt"] is None or second["lqdt"] is None:
        return []
    factor = (1 + first["twr"]) * (1 + second["twr"])
    lqdt_factor = (1 + first["lqdt"]) * (1 + second["lqdt"])
    span = first["span"] + second["span"]
    cagr = annualize(factor, span)
    lqdt_cagr = annualize(lqdt_factor, span)
    gap = None if cagr is None or lqdt_cagr is None else cagr - lqdt_cagr
    return [
        "### Оба направленных отрезка вместе",
        "",
        (
            f"Пауза с 9 марта по 24 августа не входит. Вместе отрезки длятся {span} дней. "
            f"Перемноженная доходность счета {pct(factor - 1)}, в годовых **{pct(cagr)}**. "
            f"LQDT за те же даты {pct(lqdt_factor - 1)}, в годовых {pct(lqdt_cagr)}. "
            f"Счет минус фонд, в годовых: **{pct(gap)}**."
        ),
        "",
        (
            f"Вариационная маржа по юаню за оба отрезка {rub(first['vm'] + second['vm'])}. "
            f"Комиссия за следование {rub(first['follow'] + second['follow'])}."
        ),
        "",
    ]


def brief_section(kept: dict[str, list[dict]]) -> list[str]:
    trend = kept.get("trend-yan") or []
    old = kept.get("old-good-trend") or []
    if len(trend) < 3 or len(old) < 2:
        return []
    first, second = trend[1], trend[2]
    factor = (1 + first["twr"]) * (1 + second["twr"])
    lqdt_factor = (1 + first["lqdt"]) * (1 + second["lqdt"])
    span = first["span"] + second["span"]
    cagr = annualize(factor, span)
    lqdt_cagr = annualize(lqdt_factor, span)
    directed = old[1]
    curve_t, curve_o = trend[0], old[0]
    return [
        "## Коротко",
        "",
        (
            f"Направленный режим — один истекающий фьючерс на юань. "
            f"У «Трендового юаня» два таких отрезка без долларовой связки, вместе {span} дней: "
            f"счет **{pct(cagr)}** годовых, LQDT {pct(lqdt_cagr)} годовых, "
            f"разница **{pct(cagr - lqdt_cagr)}**. "
            f"Вариационная маржа {rub(first['vm'] + second['vm'])}, комиссия за следование {rub(first['follow'] + second['follow'])}."
        ),
        "",
        (
            f"У «Старой доброй трендовушки» направленный режим идет с 7 июля 2026, {directed['span']} дней: "
            f"счет **{pct(directed['cagr'])}** годовых, LQDT {pct(directed['lqdt_cagr'])}, "
            f"разница **{pct(directed['cagr'] - directed['lqdt_cagr'])}**. "
            f"Вариационная маржа {rub(directed['vm'])}, комиссия за следование {rub(directed['follow'])}."
        ),
        "",
        (
            f"Кривая в начале 2025 — другой режим и один месяц. "
            f"«Трендовый юань»: {pct(curve_t['twr'])} за {curve_t['span']} дней, маржа {rub(curve_t['vm'])}, "
            f"комиссия брокера {rub(curve_t['broker_fee'])}. "
            f"«Старая добрая»: {pct(curve_o['twr'])} за {curve_o['span']} дней, маржа {rub(curve_o['vm'])}, "
            f"комиссия брокера {rub(curve_o['broker_fee'])}."
        ),
        "",
    ]


def main() -> None:
    books = {
        "trend-yan": build_nav("trend-yan", Path("/tmp/trend-yan-ops.json")),
        "old-good-trend": build_nav("old-good-trend", Path("/tmp/old-good-ops.json")),
    }
    # The base day is the evening before the first trade, so that day's variation
    # margin falls inside the chain. The dollar book on the trend account
    # (USDRUBF against Si, 9 March–24 August 2026) is left out.
    windows = {
        "trend-yan": [
            (
                "Кривая, февраль — начало марта 2025",
                date(2025, 2, 2),
                date(2025, 3, 4),
                "Две соседние экспирации юаня одновременно, лоты по 1–14. Бумаг на счете в эти недели не было, стоимость — это деньги. Окно захватывает утренний клиринг 4 марта, последний по этой связке.",
            ),
            (
                "Направленная торговля, 16 октября 2025 — 8 марта 2026",
                date(2025, 10, 15),
                date(2026, 3, 8),
                "Один истекающий фьючерс на юань. Денежный остаток лежит в LQDT. 9 марта на счет встала долларовая связка, поэтому отрезок обрывается накануне.",
            ),
            (
                "Направленная торговля, 25 августа — 12 октября 2026",
                date(2026, 8, 25),
                date(2026, 10, 12),
                "Долларовая связка закрыта 24 августа. С 27 августа снова один фьючерс на юань. На 12 октября позиция открыта: шорт 12 контрактов CRZ6, маржа по ней уже в деньгах по последний клиринг.",
            ),
        ],
        "old-good-trend": [
            (
                "Кривая, 22 января — 27 февраля 2025",
                date(2025, 1, 21),
                date(2025, 2, 27),
                "Пары соседних экспираций юаня, до 12 лотов на ногу. Обе ноги закрыты вечером 26 февраля, клиринг прошел 27 февраля. В те же дни на счете мелькнули газ GZM5 и мини-RTS по одному лоту.",
            ),
            (
                "Направленная торговля, 7 июля — 12 октября 2026",
                date(2026, 7, 6),
                date(2026, 10, 12),
                "Один истекающий фьючерс на юань, капитал в LQDT. На 12 октября позиция открыта: шорт 22 контракта CRZ6. Базис сентября 2023 дал по двум ногам +37 ₽ и в годовую цифру не входит.",
            ),
        ],
    }
    lines = [
        "# Доходность в годовых",
        "",
        "Счета, к которым подключены стратегии. Стоимость счета — деньги плюс рыночная цена фондов и бумаг. Вариационная маржа фьючерса каждый вечер уже падает в деньги, поэтому номинал фьючерса сверху не прибавляется.",
        "",
        "Доходность взвешена по времени: ввод, вывод и перевод между своими счетами из прибыли вынуты. Годовая цифра — сложный процент, та же доходность за период, растянутая на 365 дней. LQDT за те же даты — то, что капитал заработал бы в фонде ликвидности. Разница собирает фьючерс, комиссию брокера и комиссию за следование.",
        "",
        "На «Трендовом юане» с 9 марта по 24 августа 2026 стояла долларовая связка USDRUBF против Si. Эти месяцы в доходность юаня не входят: комиссия за следование в паузу почти не списывалась, а скачки стоимости счета в марте и апреле — маржа доллара.",
        "",
    ]
    titles = {"trend-yan": "Трендовый юань", "old-good-trend": "Старая добрая трендовушка"}
    kept: dict[str, list[dict]] = {}
    for key, book in books.items():
        lines.append(f"## {titles[key]}")
        lines.append("")
        last = max(book["nav"])
        lines.append(f"На {last.strftime('%d.%m.%Y')} стоимость счета по этой сборке {money(book['nav'][last])}.")
        lines.append("")
        kept[key] = []
        for title, start, end, note in windows[key]:
            stats = window_stats(book, start, end)
            if not stats:
                continue
            kept[key].append(stats)
            lines.extend(render_window(title, stats, note))
            if key == "trend-yan" and len(kept[key]) == 3:
                lines.extend(stitched_section(kept[key][1], kept[key][2]))
            print(
                key,
                title,
                "twr",
                pct(stats["twr"]),
                "cagr",
                pct(stats["cagr"]),
                "lqdt",
                pct(stats["lqdt_cagr"]),
                "vm",
                round(stats["vm"]),
                "other",
                round(stats["other_vm"]),
                "follow",
                round(stats["follow"]),
                "avg",
                round(stats["avg_nav"]),
                "resets",
                stats["resets"],
            )
        print(key, "last nav", round(book["nav"][last], 2), "qty", book["last_qty"])
    brief = brief_section(kept)
    insert_at = lines.index("## Трендовый юань")
    lines[insert_at:insert_at] = brief
    path = ROOT / "report" / "RETURN.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()

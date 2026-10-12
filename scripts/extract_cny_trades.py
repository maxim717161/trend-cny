#!/usr/bin/env python3
"""Extract CNY futures trades from T-Bank broker reports and a read-only API account."""

from __future__ import annotations

import csv
import json
import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cny"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
CODE_RE = re.compile(r"^CR[HMUZ]\d$|^CNYRUBF$")
CA = Path("/tmp/russian-trusted-root.pem")

COLUMNS = [
    "strategy",
    "source",
    "trade_id",
    "order_id",
    "ts",
    "side",
    "name",
    "ticker",
    "price",
    "qty",
    "value_rub",
    "exchange_fee",
]


def iter_sheet_rows(path: Path):
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
                yield texts


def parse_fill(cells: list[str]) -> dict | None:
    for i, cell in enumerate(cells):
        if cell not in ("Покупка", "Продажа") or i < 3:
            continue
        date, clock, venue = cells[i - 3], cells[i - 2], cells[i - 1]
        if not (DATE_RE.match(date) and TIME_RE.match(clock) and venue == "ММВБ"):
            continue
        if i + 4 >= len(cells):
            return None
        name, ticker, price_s, ccy, qty_s = cells[i + 1 : i + 6]
        if ccy != "RUB" or not CODE_RE.match(ticker):
            return None
        try:
            price = float(price_s)
            qty = float(qty_s)
        except ValueError:
            return None
        ts = datetime.strptime(f"{date} {clock}", "%d.%m.%Y %H:%M:%S")
        fee = 0.0
        # Exchange fee sits after settlement amount; keep the first fee-like number after qty.
        for extra in cells[i + 6 : i + 14]:
            if re.fullmatch(r"\d+\.\d+", extra) and float(extra) < 50:
                fee = float(extra)
        value = price * qty * 1000
        return {
            "trade_id": cells[0],
            "order_id": cells[1],
            "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "side": "buy" if cell == "Покупка" else "sell",
            "name": name,
            "ticker": ticker,
            "price": price,
            "qty": qty,
            "value_rub": round(value, 2),
            "exchange_fee": fee,
        }
    return None


def fills_from_reports(folder: Path, strategy: str) -> list[dict]:
    seen = set()
    fills = []
    for path in sorted(folder.glob("*.xlsx")):
        for cells in iter_sheet_rows(path):
            fill = parse_fill(cells)
            if fill is None or fill["trade_id"] in seen:
                continue
            seen.add(fill["trade_id"])
            fill["strategy"] = strategy
            fill["source"] = path.name
            fills.append(fill)
    fills.sort(key=lambda row: (row["ts"], row["trade_id"]))
    return fills


def api_post(token: str, method: str, body: dict) -> dict:
    ctx = ssl.create_default_context(cafile=str(CA))
    req = urllib.request.Request(
        "https://invest-public-api.tbank.ru/rest/" + method,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=90) as resp:
        return json.loads(resp.read().decode())


def money(quotation: dict | None) -> float:
    if not quotation:
        return 0.0
    return int(quotation.get("units") or 0) + int(quotation.get("nano") or 0) / 1e9


def fills_from_api(key_file: Path, strategy: str) -> list[dict]:
    token = key_file.read_text().strip()
    accounts = api_post(
        token,
        "tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts",
        {"status": "ACCOUNT_STATUS_ALL"},
    )["accounts"]
    account = accounts[0]
    items = []
    cursor = ""
    while True:
        page = api_post(
            token,
            "tinkoff.public.invest.api.contract.v1.OperationsService/GetOperationsByCursor",
            {
                "accountId": account["id"],
                "from": "2021-10-01T00:00:00Z",
                "to": "2026-10-13T00:00:00Z",
                "cursor": cursor,
                "limit": 1000,
                "state": "OPERATION_STATE_EXECUTED",
                "withoutCommissions": True,
                "withoutTrades": False,
                "withoutOvernights": True,
            },
        )
        items.extend(page.get("items") or [])
        if not page.get("hasNext"):
            break
        cursor = page.get("nextCursor") or ""
        if not cursor:
            break
        time.sleep(0.12)
    fills = []
    for item in items:
        ticker = item.get("ticker") or ""
        if item.get("type") not in ("OPERATION_TYPE_BUY", "OPERATION_TYPE_SELL"):
            continue
        if not CODE_RE.match(ticker):
            continue
        ts = datetime.fromisoformat(item["date"].replace("Z", "+00:00")) + timedelta(hours=3)
        trades = ((item.get("tradesInfo") or {}).get("trades")) or [{}]
        # One broker operation is already one fill in this API.
        fills.append(
            {
                "strategy": strategy,
                "source": "api:" + account["name"],
                "trade_id": item["id"],
                "order_id": item.get("parentOperationId") or "",
                "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "side": "buy" if item["type"] == "OPERATION_TYPE_BUY" else "sell",
                "name": item.get("name") or "",
                "ticker": ticker,
                "price": money(item.get("price")),
                "qty": float(item.get("quantity") or 0),
                "value_rub": round(money(item.get("price")) * float(item.get("quantity") or 0) * 1000, 2),
                "exchange_fee": 0.0,
            }
        )
    fills.sort(key=lambda row: (row["ts"], row["trade_id"]))
    return fills


def group_orders(fills: list[dict]) -> list[dict]:
    """Collapse iceberg slices of the same contract and side into one decision."""
    orders = []
    current = None
    for fill in fills:
        ts = datetime.strptime(fill["ts"], "%Y-%m-%d %H:%M:%S")
        if (
            current
            and current["ticker"] == fill["ticker"]
            and current["side"] == fill["side"]
            and (ts - current["end"]) <= timedelta(minutes=5)
        ):
            current["end"] = ts
            current["qty"] += fill["qty"]
            current["notional"] += fill["price"] * fill["qty"]
            current["value_rub"] += fill["value_rub"]
            current["n_fills"] += 1
            continue
        if current:
            orders.append(finish_order(current))
        current = {
            "strategy": fill["strategy"],
            "ticker": fill["ticker"],
            "name": fill["name"],
            "side": fill["side"],
            "start": ts,
            "end": ts,
            "qty": fill["qty"],
            "notional": fill["price"] * fill["qty"],
            "value_rub": fill["value_rub"],
            "n_fills": 1,
        }
    if current:
        orders.append(finish_order(current))
    # Position after each order, per contract.
    pos = defaultdict(float)
    for order in orders:
        signed_qty = lots(order["qty"] if order["side"] == "buy" else -order["qty"])
        before = lots(pos[order["ticker"]])
        after = lots(before + signed_qty)
        pos[order["ticker"]] = after
        order["pos_before"] = before
        order["pos_after"] = after
        order["kind"] = kind_of(before, after)
    return orders


def finish_order(raw: dict) -> dict:
    return {
        "strategy": raw["strategy"],
        "ticker": raw["ticker"],
        "name": raw["name"],
        "side": raw["side"],
        "ts": raw["start"].strftime("%Y-%m-%d %H:%M:%S"),
        "ts_end": raw["end"].strftime("%Y-%m-%d %H:%M:%S"),
        "qty": raw["qty"],
        "price": round(raw["notional"] / raw["qty"], 6) if raw["qty"] else 0,
        "value_rub": round(raw["value_rub"], 2),
        "n_fills": raw["n_fills"],
    }


def kind_of(before: float, after: float) -> str:
    if before == 0 and after > 0:
        return "enter_long"
    if before == 0 and after < 0:
        return "enter_short"
    if after == 0 and before > 0:
        return "exit_long"
    if after == 0 and before < 0:
        return "exit_short"
    if after > before > 0:
        return "add_long"
    if after < before < 0:
        return "add_short"
    if before > after > 0:
        return "reduce_long"
    if before < after < 0:
        return "reduce_short"
    if before > 0 > after:
        return "flip_short"
    if before < 0 < after:
        return "flip_long"
    return "adjust"


def lots(value: float) -> float:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-6:
        return float(rounded)
    return float(value)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fills_after(key_file: Path, strategy: str, after_ts: str) -> list[dict]:
    """CNY fills on the key's account strictly later than an Excel cutoff."""
    token = key_file.read_text().strip()
    accounts = api_post(
        token,
        "tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts",
        {"status": "ACCOUNT_STATUS_ALL"},
    )["accounts"]
    account = accounts[0]
    start = datetime.strptime(after_ts, "%Y-%m-%d %H:%M:%S") - timedelta(hours=3)
    items = []
    cursor = ""
    while True:
        page = api_post(
            token,
            "tinkoff.public.invest.api.contract.v1.OperationsService/GetOperationsByCursor",
            {
                "accountId": account["id"],
                "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": "2026-10-13T00:00:00Z",
                "cursor": cursor,
                "limit": 1000,
                "state": "OPERATION_STATE_EXECUTED",
                "withoutCommissions": True,
                "withoutTrades": False,
                "withoutOvernights": True,
            },
        )
        items.extend(page.get("items") or [])
        if not page.get("hasNext"):
            break
        cursor = page.get("nextCursor") or ""
        if not cursor:
            break
        time.sleep(0.12)
    fills = []
    for item in items:
        ticker = item.get("ticker") or ""
        if item.get("type") not in ("OPERATION_TYPE_BUY", "OPERATION_TYPE_SELL"):
            continue
        if not CODE_RE.match(ticker):
            continue
        ts = datetime.fromisoformat(item["date"].replace("Z", "+00:00")) + timedelta(hours=3)
        stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
        if stamp <= after_ts:
            continue
        price = money(item.get("price"))
        qty = lots(item.get("quantity") or 0)
        fills.append(
            {
                "strategy": strategy,
                "source": "api:" + account["name"],
                "trade_id": item["id"],
                "order_id": item.get("parentOperationId") or "",
                "ts": stamp,
                "side": "buy" if item["type"] == "OPERATION_TYPE_BUY" else "sell",
                "name": item.get("name") or "",
                "ticker": ticker,
                "price": price,
                "qty": qty,
                "value_rub": round(price * qty * 1000, 2),
                "exchange_fee": 0.0,
            }
        )
    fills.sort(key=lambda row: (row["ts"], row["trade_id"]))
    return fills


def futures_position(key_file: Path) -> tuple[str, list[tuple[str, float]]]:
    token = key_file.read_text().strip()
    account = api_post(
        token,
        "tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts",
        {"status": "ACCOUNT_STATUS_ALL"},
    )["accounts"][0]
    portfolio = api_post(
        token,
        "tinkoff.public.invest.api.contract.v1.OperationsService/GetPortfolio",
        {"accountId": account["id"]},
    )
    held = []
    for position in portfolio.get("positions") or []:
        ticker = position.get("ticker") or ""
        if not CODE_RE.match(ticker):
            continue
        qty = lots(money(position.get("quantity")))
        if qty:
            held.append((ticker, qty))
    return account["name"], held


def build_campaigns(orders: list[dict]) -> list[dict]:
    """Flat-to-flat campaigns. Price PnL uses average cost and 1000 RUB per point."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for order in orders:
        grouped[order["ticker"]].append(order)
    campaigns = []
    for ticker, rows in grouped.items():
        pos = 0.0
        avg = 0.0
        camp = None
        realized = 0.0

        def start(direction: str, ts: str, qty: float, price: float) -> dict:
            return {
                "strategy": rows[0]["strategy"],
                "ticker": ticker,
                "direction": direction,
                "entry_ts": ts,
                "exit_ts": "",
                "entry_qty": qty,
                "max_qty": qty,
                "end_pos": qty if direction == "long" else -qty,
                "entry_price": price,
                "exit_price": "",
                "n_orders": 1,
                "n_add": 0,
                "n_reduce": 0,
                "path": "0→" + signed(qty if direction == "long" else -qty),
                "status": "open",
            }

        def finish(ts: str, status: str, price: float, end_pos: float) -> None:
            nonlocal camp, realized
            camp["exit_ts"] = ts
            camp["status"] = status
            camp["exit_price"] = price
            camp["pnl_rub"] = round(realized, 2)
            camp["end_pos"] = end_pos
            opened = datetime.strptime(camp["entry_ts"], "%Y-%m-%d %H:%M:%S")
            closed = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            camp["hold_hours"] = round((closed - opened).total_seconds() / 3600, 1)
            camp["phase"] = phase_of(camp["strategy"], ticker, opened)
            campaigns.append(camp)
            camp = None

        for order in rows:
            qty = lots(order["qty"])
            price = float(order["price"])
            if order["side"] == "buy":
                close_qty = min(qty, max(-pos, 0.0))
                open_qty = qty - close_qty
            else:
                close_qty = min(qty, max(pos, 0.0))
                open_qty = qty - close_qty
            if close_qty > 1e-9:
                realized += ((price - avg) if pos > 0 else (avg - price)) * close_qty * 1000
                pos = lots(pos + close_qty if pos < 0 else pos - close_qty)
                camp["n_orders"] += 1
                camp["path"] += "→" + signed(pos)
                if abs(pos) < 1e-6:
                    pos = 0.0
                    finish(order["ts"], "closed", price, 0.0)
                else:
                    camp["n_reduce"] += 1
                    camp["max_qty"] = max(camp["max_qty"], abs(pos))
            if open_qty > 1e-9:
                if abs(pos) < 1e-6:
                    direction = "long" if order["side"] == "buy" else "short"
                    camp = start(direction, order["ts"], open_qty, price)
                    avg = price
                    pos = open_qty if direction == "long" else -open_qty
                    realized = 0.0
                else:
                    abs_pos = abs(pos)
                    avg = (avg * abs_pos + price * open_qty) / (abs_pos + open_qty)
                    pos = lots(pos + open_qty if pos > 0 else pos - open_qty)
                    camp["n_add"] += 1
                    camp["n_orders"] += 1
                    camp["max_qty"] = max(camp["max_qty"], abs(pos))
                    camp["path"] += "→" + signed(pos)
        if camp:
            saved = realized
            finish(rows[-1]["ts"], "open", rows[-1]["price"], pos)
            campaigns[-1]["pnl_rub"] = round(saved, 2)
            campaigns[-1]["end_pos"] = pos
    return campaigns


def signed(qty: float) -> str:
    n = int(round(qty))
    if n > 0:
        return f"+{n}"
    return str(n)


def phase_of(strategy: str, ticker: str, entry: datetime) -> str:
    if strategy == "old-good-trend" and ticker in {"CRZ3", "CNYRUBF"}:
        return "basis"
    if strategy == "trend-yan" and entry < datetime(2025, 10, 1):
        return "curve"
    if strategy == "old-good-trend" and entry < datetime(2026, 1, 1):
        return "curve"
    return "directional"


def contract_rows(fills: list[dict], orders: list[dict]) -> list[dict]:
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for order in orders:
        by_ticker[order["ticker"]].append(order)
    out = []
    for ticker, rows in by_ticker.items():
        buys = sum(lots(r["qty"]) for r in rows if r["side"] == "buy")
        sells = sum(lots(r["qty"]) for r in rows if r["side"] == "sell")
        out.append(
            {
                "ticker": ticker,
                "name": rows[0]["name"],
                "orders": len(rows),
                "fills": sum(1 for f in fills if f["ticker"] == ticker),
                "first": rows[0]["ts"][:10],
                "last": rows[-1]["ts"][:10],
                "buy": buys,
                "sell": sells,
                "end": lots(rows[-1]["pos_after"]),
                "max_abs": max(abs(lots(r["pos_after"])) for r in rows),
            }
        )
    return out


CAMPAIGN_FIELDS = [
    "strategy",
    "phase",
    "ticker",
    "direction",
    "status",
    "entry_ts",
    "exit_ts",
    "hold_hours",
    "entry_qty",
    "max_qty",
    "end_pos",
    "entry_price",
    "exit_price",
    "n_orders",
    "n_add",
    "n_reduce",
    "pnl_rub",
    "path",
]


def fmt_int(value: float) -> str:
    return str(int(round(value)))


def fmt_pnl(value: float) -> str:
    n = int(round(value))
    body = f"{abs(n):,}".replace(",", " ")
    if n > 0:
        return "+" + body
    if n < 0:
        return "−" + body
    return "0"


def fmt_px(value: float | str) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def campaign_table(camps: list[dict]) -> str:
    rows = []
    for camp in camps:
        closed = camp["status"] == "closed"
        status = "закрыта" if closed else f"открыта {signed(camp['end_pos'])}"
        pnl = fmt_pnl(camp["pnl_rub"]) if closed or camp["n_reduce"] else "—"
        rows.append(
            [
                camp["entry_ts"][:16],
                camp["exit_ts"][:16] if closed else "—",
                camp["ticker"],
                "лонг" if camp["direction"] == "long" else "шорт",
                fmt_int(camp["entry_qty"]),
                fmt_int(camp["max_qty"]),
                fmt_px(camp["entry_price"]),
                fmt_px(camp["exit_price"]) if closed else "—",
                f"{camp['hold_hours']:.0f}" if closed else "—",
                pnl,
                status,
                camp["path"],
            ]
        )
    return md_table(
        ["Вход", "Выход", "Контракт", "Сторона", "Лоты", "Макс", "Цена входа", "Цена выхода", "Часы", "PnL, ₽", "Статус", "Позиция"],
        rows,
    )


def summarize(camps: list[dict]) -> str:
    closed = [c for c in camps if c["status"] == "closed"]
    if not closed:
        return "закрытых кампаний нет"
    pnl = sum(c["pnl_rub"] for c in closed)
    wins = sum(c["pnl_rub"] > 0 for c in closed)
    holds = sorted(c["hold_hours"] for c in closed)
    median = holds[len(holds) // 2]
    return (
        f"{len(closed)} закрытых, ценовой результат {fmt_pnl(pnl)} ₽, "
        f"прибыльных {wins} из {len(closed)}, медиана удержания {median:.0f} ч"
    )


def write_report(books: dict, positions: dict, generated_at: str) -> None:
    trend = books["trend-yan"]
    old = books["old-good-trend"]
    lines = [
        "# Сделки фьючерсом на юань",
        "",
        f"Срез на {generated_at} МСК. Время сделок — московское. Цена контракта в рублях за юань, шаг 1,0 стоит 1000 ₽ на один контракт. Результат кампании — вариационная маржа по средней цене, без биржевой и брокерской комиссии.",
        "",
        "В шапке отчета «Трендовый юань» указан счет 2167722079 от 18.04.2022 — тот же подключенный счет, что открывает ключ `key/trend-yan.txt` (только чтение). В шапке отчетов «Старая добрая трендовушка» указан счет 2116402021 от 21.10.2021 — тот же счет, что открывает ключ `key/old-good-trend.txt`.",
        "",
        "## Как собрана лента",
        "",
        "Строка сделки в Excel — это ячейка «Покупка» или «Продажа», перед которой стоят дата, время и площадка «ММВБ», а следом наименование, код `CR*` или `CNYRUBF`, цена и количество. В отчетах 2026 года между номером поручения и датой стоит признак «А» и постоянный номер мандата. В феврале 2025 этого столбца нет, номер поручения меняется на каждую сделку. Пустые ячейки в файле выкинуты, поэтому колонка ищется по содержимому, а не по номеру.",
        "",
        "Соседние исполнения одного контракта и одной стороны с разрывом не больше 5 минут склеены в одну заявку: брокер режет крупный приказ на лоты по 1–3 контракта. Кампания — путь позиции по одному контракту от нуля до нуля. Переворот через ноль закрывает одну кампанию и открывает следующую по цене этой заявки. Частичное сокращение остается внутри кампании. У еще открытой кампании в колонке PnL стоит только уже зафиксированный результат сокращений; если сокращений не было, стоит прочерк.",
        "",
        "Файлы: `data/cny/trend-yan-fills.csv`, `data/cny/trend-yan-orders.csv`, `data/cny/trend-yan-campaigns.csv` и те же три для `old-good-trend`.",
        "",
        "## Трендовый юань",
        "",
        "Источник истории до 7 октября 2026 включительно — пять брокерских отчетов `broker-report/trend-yan/`. В файлах 2022, 2023 и 2024 годов сделок с фьючерсом на юань нет. Первая сделка — 3 февраля 2025, 23:33, CRZ5. Отчет 2026 года заканчивается 7 октября. Пять продаж CRZ6 8 октября (12 контрактов, 14:41–14:44) дописаны из ключа: в Excel их еще нет. После 8 октября новых сделок по юаню ключ не показывает.",
        "",
        f"Сейчас на счете: {format_positions(positions['trend-yan'])}.",
        "",
        f"Исполнений {len(trend['fills'])}, заявок {len(trend['orders'])}, кампаний {len(trend['campaigns'])}.",
        "",
        md_table(
            ["Контракт", "С", "По", "Заявки", "Сделки", "Куплено", "Продано", "Остаток", "Пик"],
            [
                [
                    row["ticker"],
                    row["first"],
                    row["last"],
                    str(row["orders"]),
                    str(row["fills"]),
                    fmt_int(row["buy"]),
                    fmt_int(row["sell"]),
                    signed(row["end"]),
                    fmt_int(row["max_abs"]),
                ]
                for row in trend["contracts"]
            ],
        ),
        "",
        "### Кривая, февраль — начало марта 2025",
        "",
        summarize(trend["curve"]) + ".",
        "",
        "Счет одновременно держит разные экспирации. С 4 по 24 февраля лонг CRM5 идет вместе с шортом CRU5: набор и сброс почти одновременные, пик 14 контрактов на ногу. CRZ5 в те же дни то присоединяется к этой паре, то торгуется отдельно короткими заходами по 1–6 контрактов. Размер заявки в эту фазу — 1–3 лота, часы размазаны по всей сессии, включая вечер. Это календарный спред.",
        "",
        campaign_table(trend["curve"]),
        "",
        "### Один контракт, с 16 октября 2025",
        "",
        summarize(trend["directional_closed"]) + ".",
        "",
        "Дальше в книге один истекающий контракт. Базовый размер вырос ступеньками: 4 лота 16 октября, затем 8–9, с сентября 2026 года 12. До двойного размера позиция доводилась трижды: 4 марта 8→17, 9 сентября 12→25, 2 октября 12→27. Вход чаще всего в час 13:00, удержание около суток, выход часто на следующее утро в 10–11 часов или днем после 13:00. 27 августа 2026 счет вошел сразу 25 лотами в 17:12 и на следующий день сократил позицию до 12, затем закрыл.",
        "",
        campaign_table(trend["directional"]),
        "",
        "## Старая добрая трендовушка",
        "",
        "Источник — шесть брокерских отчетов `broker-report/old-good-trend/` с 21 октября 2021 по 7 октября 2026. В файлах 2021, 2022 и 2024 годов сделок с фьючерсом на юань нет. Первая сделка — 12 сентября 2023. Между ноябрем 2023 и январем 2025, и между мартом 2025 и июлем 2026 сделок с юанем нет. Отчет 2026 года заканчивается 7 октября; сделки 8, 9 и 12 октября дописаны из ключа.",
        "",
        f"Сейчас на счете: {format_positions(positions['old-good-trend'])}.",
        "",
        f"Исполнений {len(old['fills'])}, заявок {len(old['orders'])}, кампаний {len(old['campaigns'])}.",
        "",
        md_table(
            ["Контракт", "С", "По", "Заявки", "Сделки", "Куплено", "Продано", "Остаток", "Пик"],
            [
                [
                    row["ticker"],
                    row["first"],
                    row["last"],
                    str(row["orders"]),
                    str(row["fills"]),
                    fmt_int(row["buy"]),
                    fmt_int(row["sell"]),
                    signed(row["end"]),
                    fmt_int(row["max_abs"]),
                ]
                for row in old["contracts"]
            ],
        ),
        "",
        "### Базис, сентябрь — октябрь 2023",
        "",
        summarize(old["basis"]) + ".",
        "",
        "Короткий шорт квартального CRZ3 против лонга вечного CNYRUBF, по 3 контракта, удержание около месяца. Ноги закрыты 11 октября 2023 в одну минуту.",
        "",
        campaign_table(old["basis"]),
        "",
        "### Кривая, 22 января — 26 февраля 2025",
        "",
        summarize(old["curve"]) + ".",
        "",
        "Четыре экспирации сразу: CRM5, CRH5, CRZ5 и CRU5. 22–24 января это пара шорт CRM5 / лонг CRH5 по 10 контрактов. С 27 января добавляется шорт CRZ5 против лонга CRM5. С 31 января по 4 февраля лонг CRU5 против шорта CRH5. С 10 по 20 февраля пара развернута в другую сторону, пик 12 контрактов на ногу. 24–26 февраля снова лонг CRU5 и шорт CRH5 по 12. Вечером 26 февраля обе ноги закрыты: в разделе позиций на расчетную дату 27 февраля по CRU5 и CRH5 стоит ноль. Исполнения в Excel мельче, чем операции ключа, и по CRU5 количества не совпадали. Лента ниже собрана из отчета.",
        "",
        campaign_table(old["curve"]),
        "",
        "### Один контракт, июль — октябрь 2026",
        "",
        summarize(old["directional_closed"]) + ".",
        "",
        "С 7 июля 2026 счет торгует один контракт: сначала CRU6, с 21 сентября CRZ6. Вход почти всегда в час 13:00, часто несколькими заявками в ту же сессию, до 36 и один раз до 57 контрактов. Сокращение на следующий день идет несколькими заявками. В те же дни размер здесь в разы больше, чем на «Трендовом юане».",
        "",
        campaign_table(old["directional"]),
        "",
        "## Где направление совпало",
        "",
        "Календари разные. «Трендовый юань» направленно торгует с октября 2025, «Старая добрая трендовушка» в эту зиму молчит и возвращается только 7 июля 2026. Общие входы в один контракт в пределах шести часов:",
        "",
        same_day_table(trend["campaigns"], old["campaigns"]),
        "",
        "Сторона в эти дни одна. Размер и минута входа различаются. В феврале 2025 оба счета крутят календарную кривую, набор экспираций при этом разный: у первого CRM5/CRU5/CRZ5, у второго CRH5/CRU5/CRM5/CRZ5. Это две похожие книги разного размера.",
        "",
    ]
    path = ROOT / "report" / "CNY-TRADES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("report", path)


def same_day_table(left: list[dict], right: list[dict]) -> str:
    rows = []
    for a in left:
        for b in right:
            if a["ticker"] != b["ticker"] or a["phase"] != "directional" or b["phase"] != "directional":
                continue
            dt_hours = (
                datetime.strptime(a["entry_ts"], "%Y-%m-%d %H:%M:%S")
                - datetime.strptime(b["entry_ts"], "%Y-%m-%d %H:%M:%S")
            ).total_seconds() / 3600
            if abs(dt_hours) > 6:
                continue
            rows.append(
                [
                    a["entry_ts"][:10],
                    a["ticker"],
                    ("лонг" if a["direction"] == "long" else "шорт"),
                    a["entry_ts"][11:16] + f" / {fmt_int(a['entry_qty'])}→{fmt_int(a['max_qty'])}",
                    b["entry_ts"][11:16] + f" / {fmt_int(b['entry_qty'])}→{fmt_int(b['max_qty'])}",
                    f"{dt_hours:+.1f} ч",
                ]
            )
    if not rows:
        return "Общих направленных входов нет."
    return md_table(
        ["Дата", "Контракт", "Сторона", "Трендовый юань", "Старая добрая", "Разница входа"],
        rows,
    )


def format_positions(held: list[tuple[str, float]]) -> str:
    if not held:
        return "фьючерса на юань нет"
    parts = []
    for ticker, qty in held:
        side = "лонг" if qty > 0 else "шорт"
        parts.append(f"{side} {fmt_int(abs(qty))} {ticker}")
    return ", ".join(parts)


def load_book(name: str, fills: list[dict]) -> dict:
    orders = group_orders(fills)
    campaigns = build_campaigns(orders)
    book = {
        "fills": fills,
        "orders": orders,
        "campaigns": campaigns,
        "contracts": contract_rows(fills, orders),
        "curve": [c for c in campaigns if c["phase"] == "curve"],
        "basis": [c for c in campaigns if c["phase"] == "basis"],
        "directional": [c for c in campaigns if c["phase"] == "directional"],
        "directional_closed": [
            c for c in campaigns if c["phase"] == "directional" and c["status"] == "closed"
        ],
    }
    write_csv(OUT / f"{name}-fills.csv", fills, COLUMNS)
    write_csv(
        OUT / f"{name}-orders.csv",
        orders,
        [
            "strategy",
            "ticker",
            "name",
            "side",
            "ts",
            "ts_end",
            "qty",
            "price",
            "value_rub",
            "n_fills",
            "pos_before",
            "pos_after",
            "kind",
        ],
    )
    write_csv(OUT / f"{name}-campaigns.csv", campaigns, CAMPAIGN_FIELDS)
    from collections import Counter

    print(name, "fills", len(fills), "orders", len(orders), "campaigns", len(campaigns))
    print(" ", Counter(o["kind"] for o in orders))
    return book


def main() -> None:
    trend = fills_from_reports(ROOT / "broker-report" / "trend-yan", "trend-yan")
    extra = fills_after(ROOT / "key" / "trend-yan.txt", "trend-yan", trend[-1]["ts"])
    print("trend-yan excel", len(trend), "api tail", len(extra))
    trend.extend(extra)
    old = fills_from_reports(ROOT / "broker-report" / "old-good-trend", "old-good-trend")
    old_extra = fills_after(ROOT / "key" / "old-good-trend.txt", "old-good-trend", old[-1]["ts"])
    print("old-good-trend excel", len(old), "api tail", len(old_extra))
    old.extend(old_extra)
    books = {
        "trend-yan": load_book("trend-yan", trend),
        "old-good-trend": load_book("old-good-trend", old),
    }
    positions = {
        "trend-yan": futures_position(ROOT / "key" / "trend-yan.txt")[1],
        "old-good-trend": futures_position(ROOT / "key" / "old-good-trend.txt")[1],
    }
    print("positions", positions)
    write_report(books, positions, "12 октября 2026")


if __name__ == "__main__":
    main()

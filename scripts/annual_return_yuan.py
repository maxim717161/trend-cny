#!/usr/bin/env python3
"""Annual yield of the two accounts over the time they actually trade the yuan future."""

from __future__ import annotations

import csv
import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("annual_return", ROOT / "scripts" / "annual_return.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)

END = date(2026, 10, 12)
TITLES = {"trend-yan": "Трендовый юань", "old-good-trend": "Старая добрая трендовушка"}
# The 2023 basis was 3 lots, +37 ₽ in total, and the account held other books then.
FIRST_DAY = {"trend-yan": date(2025, 2, 3), "old-good-trend": date(2025, 1, 22)}


def active_days(folder: str) -> set[date]:
    """Days with a yuan position open from entry to the clearing after the exit."""
    days: set[date] = set()
    with (ROOT / "data" / "cny" / f"{folder}-campaigns.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            entry = datetime.fromisoformat(row["entry_ts"]).date()
            if row["status"] == "open":
                exit_day = END
            else:
                exit_day = datetime.fromisoformat(row["exit_ts"]).date() + timedelta(days=1)
            if entry < FIRST_DAY[folder]:
                continue
            day = entry
            while day <= min(exit_day, END):
                days.add(day)
                day += timedelta(days=1)
    return days


def chain(book: dict, days: set[date]) -> dict:
    factor = 1.0
    lqdt_factor = 1.0
    lqdt = book["prices"].get("LQDT", {})
    count = 0
    vm = 0.0
    follow = 0.0
    broker_fee = 0.0
    navs = []
    jumps = []
    for day in sorted(days):
        before = day - timedelta(days=1)
        if day not in book["nav"] or before not in book["nav"]:
            continue
        prev = book["nav"][before]
        if prev < 1000:
            continue
        flow = book["flows"].get(day, 0.0)
        growth = (book["nav"][day] - flow) / prev
        factor *= growth
        if abs(growth - 1) > 0.08:
            jumps.append((day, growth - 1))
        if lqdt.get(day) and lqdt.get(before):
            lqdt_factor *= lqdt[day] / lqdt[before]
        count += 1
        navs.append(prev)
        vm += book["vm"].get(day, 0.0)
        follow += book["follow"].get(day, 0.0)
        broker_fee += book["broker_fee"].get(day, 0.0)
    avg = sum(navs) / len(navs) if navs else 0.0
    return {
        "days": count,
        "factor": factor,
        "twr": factor - 1,
        "cagr": ar.annualize(factor, count),
        "lqdt": lqdt_factor - 1,
        "lqdt_cagr": ar.annualize(lqdt_factor, count),
        "vm": vm,
        "follow": follow,
        "broker_fee": broker_fee,
        "avg_nav": avg,
        "vm_on_nav": vm / avg if avg else 0.0,
        "vm_annual": (vm / avg) * (365.25 / count) if avg and count else 0.0,
        "jumps": jumps,
    }


def main() -> None:
    books = {
        "trend-yan": ar.build_nav("trend-yan", Path("/tmp/trend-yan-ops.json")),
        "old-good-trend": ar.build_nav("old-good-trend", Path("/tmp/old-good-ops.json")),
    }
    lines = [
        "# Годовая доходность за время торговли юанем",
        "",
        "Считается по счетам из брокерских отчетов. Кэш и вводы сверены с отчетами, бумаги и фонд LQDT оценены по закрытиям Мосбиржи. Вариационная маржа по юаню взята из строк отчета с кодом `CNY-…`, хвост после 7 октября 2026 — из API.",
        "",
        "Два способа считать. Первый — календарь от первой сделки по юаню до 12 октября 2026, все дни подряд. Второй — только дни, когда по юаню была открыта позиция, от входа до клиринга после выхода. Во втором доходность — произведение дневных доходностей этих дней, годовая цифра растянута на 365 дней по числу этих дней.",
        "",
    ]
    for key in ("trend-yan", "old-good-trend"):
        book = books[key]
        start = FIRST_DAY[key]
        calendar = ar.window_stats(book, start - timedelta(days=1), END)
        days = active_days(key)
        active = chain(book, days)
        lines.append(f"## {TITLES[key]}")
        lines.append("")
        lines.append(
            f"Первая сделка по юаню в окне — {start.strftime('%d.%m.%Y')}. "
            f"Дней с открытой позицией по юаню: {len(days)} из {(END - start).days + 1}."
        )
        lines.append("")
        lines.append("### Календарь, все дни")
        lines.append("")
        if key == "trend-yan":
            lines.append(
                "В эти дни входит долларовая связка USDRUBF против Si с 9 марта по 24 августа 2026: она дала основную часть прироста и все скачки ниже. Это не юань."
            )
            lines.append("")
        lines.append(
            f"С {calendar['start'].strftime('%d.%m.%Y')} по {calendar['end'].strftime('%d.%m.%Y')}, {calendar['span']} дней. "
            f"Стоимость счета {ar.money(calendar['nav0'])} → {ar.money(calendar['nav1'])}. "
            f"Вводы минус выводы {ar.rub(calendar['deposits'])}. Прирост после вводов {ar.rub(calendar['profit'])}."
        )
        lines.append("")
        lines.append(f"- Доходность счета за период: **{ar.pct(calendar['twr'])}**, в годовых **{ar.pct(calendar['cagr'])}**.")
        lines.append(
            f"- LQDT за те же даты: {ar.pct(calendar['lqdt'])}, в годовых {ar.pct(calendar['lqdt_cagr'])}."
        )
        if calendar["resets"]:
            lines.append(
                f"- Дней, когда счет стоял пустым после вывода, пока его не пополнили: {calendar['resets']}. "
                f"Доходность на них не набегает, пополнение считается новой базой."
            )
        lines.append(
            f"- Вариационная маржа по юаню: {ar.rub(calendar['vm'])}, это {ar.pct(calendar['vm_on_nav'])} от средней стоимости {ar.money(calendar['avg_nav'])}."
        )
        lines.append(f"- Комиссия за следование: {ar.rub(calendar['follow'])}, комиссия брокера: {ar.rub(calendar['broker_fee'])}.")
        if calendar["jumps"]:
            lines.append("- Сдвиги больше 8% за день, не связанные с вводами:")
            for day, ret, flow, before, after in calendar["jumps"][:12]:
                lines.append(f"  - {day.strftime('%d.%m.%Y')}: {ar.pct(ret)}, счет {ar.money(before)} → {ar.money(after)}.")
        lines.append("")
        lines.append("### Только дни с позицией по юаню")
        lines.append("")
        lines.append(
            f"{active['days']} дней. Доходность за эти дни: **{ar.pct(active['twr'])}**, в годовых **{ar.pct(active['cagr'])}**. "
            f"LQDT за те же дни: {ar.pct(active['lqdt'])}, в годовых {ar.pct(active['lqdt_cagr'])}."
        )
        lines.append("")
        lines.append(
            f"Вариационная маржа по юаню за эти дни {ar.rub(active['vm'])}, это {ar.pct(active['vm_on_nav'])} от средней стоимости "
            f"{ar.money(active['avg_nav'])}, в годовых без сложного процента {ar.pct(active['vm_annual'])}."
        )
        lines.append("")
        if active["jumps"]:
            lines.append("Сдвиги больше 8% за день внутри этих дней:")
            for day, ret in active["jumps"][:12]:
                lines.append(f"- {day.strftime('%d.%m.%Y')}: {ar.pct(ret)}.")
            lines.append("")
        print(
            key,
            "calendar twr",
            ar.pct(calendar["twr"]),
            "cagr",
            ar.pct(calendar["cagr"]),
            "lqdt",
            ar.pct(calendar["lqdt_cagr"]),
            "span",
            calendar["span"],
            "resets",
            calendar["resets"],
            "| active days",
            active["days"],
            "twr",
            ar.pct(active["twr"]),
            "cagr",
            ar.pct(active["cagr"]),
            "lqdt",
            ar.pct(active["lqdt_cagr"]),
            "vm",
            round(active["vm"]),
            "avg",
            round(active["avg_nav"]),
            "vm_annual",
            ar.pct(active["vm_annual"]),
        )
    path = ROOT / "report" / "RETURN-YUAN.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()

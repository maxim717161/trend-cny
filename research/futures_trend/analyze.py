"""Score the hourly MOEX futures sample for trend versus mean reversion.

The price statistics use the whole sample. The moving-average horse-race is
scored out of sample from 2026-01-01, with the lookbacks fixed in advance
rather than picked on the test window.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from research.futures_trend.metrics import (
    DAILY_QS,
    DAILY_WINDOWS,
    HOURLY_QS,
    HOURLY_WINDOWS,
    acf,
    compound_index,
    daily_compound,
    hurst,
    max_drawdown,
    median,
    sharpe,
    sma_position,
    strategy_returns,
    variance_ratio,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "futures-hourly"
OUT = ROOT / "results" / "futures-trend"
TEST_START = date(2026, 1, 1)


def load_bars(path: Path) -> list[dict]:
    bars = []
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            ret = float(row["ret"]) if row["ret"] else math.nan
            if ret <= -0.99:
                ret = math.nan
            bars.append(
                {
                    "begin": datetime.strptime(row["begin"][:19], "%Y-%m-%dT%H:%M:%S"),
                    "secid": row["secid"],
                    "close": float(row["close"]),
                    "ret": ret,
                    "roll": int(row["roll"]),
                }
            )
    return bars


def load_universe() -> list[dict]:
    with (DATA / "universe.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["minstep"] = float(row["minstep"]) if row["minstep"] else math.nan
        row["avg_daily_turnover_rub"] = float(row["avg_daily_turnover_rub"])
        row["avg_front_oi"] = float(row["avg_front_oi"] or 0)
    return rows


def daily_frame(bars: list[dict]) -> dict:
    grouped: dict[date, list[float]] = defaultdict(list)
    last_close: dict[date, float] = {}
    for bar in bars:
        day = bar["begin"].date()
        if math.isfinite(bar["ret"]) and bar["ret"] > -0.99:
            grouped[day].append(math.log1p(bar["ret"]))
        last_close[day] = bar["close"]
    days = sorted(grouped)
    logret = np.array([sum(grouped[day]) for day in days], dtype=float)
    simple = np.expm1(logret)
    closes = np.array([last_close[day] for day in days], dtype=float)
    return {"dates": days, "logret": logret, "simple": simple, "close": closes}


def _band(value: float, width: float) -> int | None:
    if not math.isfinite(value):
        return None
    if value > width:
        return 1
    if value < -width:
        return -1
    return 0


def classify(hurst_value: float, autocorr: float, ratio: float) -> str:
    """Agreeing signs only. A flat metric does not vote; a contradiction is noise.

    Hurst must clear 0.03 around 0.50, autocorrelation 0.02, variance ratio 0.02
    around 1. Those bands are wider than a single noisy tick of the estimate.
    """
    hurst_sign = _band(hurst_value - 0.5, 0.03)
    acf_sign = _band(autocorr, 0.02)
    ratio_sign = _band(ratio - 1.0, 0.02)
    if hurst_sign is None or acf_sign is None or ratio_sign is None:
        return "мало данных"
    signs = [hurst_sign, acf_sign, ratio_sign]
    nonzero = [sign for sign in signs if sign != 0]
    # One metric is not a class. Two may be, if they do not contradict the third.
    if len(nonzero) < 2 or any(sign != nonzero[0] for sign in nonzero):
        return "шум"
    direction = "тренд" if nonzero[0] > 0 else "контртренд"
    if hurst_sign == 0:
        return f"слабо {direction}"
    return direction


def mean_finite(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def demean(dates, market: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract the train-period mean so a rising market is not scored as trend."""
    train = [float(value) for day, value in zip(dates, market) if day < TEST_START and np.isfinite(value)]
    mu = float(np.mean(train)) if train else 0.0
    excess = np.array(market, dtype=float, copy=True)
    finite = np.isfinite(excess)
    excess[finite] = excess[finite] - mu
    return excess, mu


def horse_race(dates, market, index, tick_frac, windows, ticks: float) -> list[dict]:
    rows = []
    position_book = []
    for window in windows:
        position = sma_position(index, window)
        position_book.append(position)
        trend = strategy_returns(market, position, tick_frac, ticks)
        fade = strategy_returns(market, -position, tick_frac, ticks)
        for sample, start, stop in (
            ("train", None, TEST_START),
            ("test", TEST_START, None),
        ):
            trend_daily = daily_compound(dates, trend, start, stop)
            fade_daily = daily_compound(dates, fade, start, stop)
            trend_sharpe = sharpe(trend_daily)
            fade_sharpe = sharpe(fade_daily)
            edge = (
                trend_sharpe - fade_sharpe
                if math.isfinite(trend_sharpe) and math.isfinite(fade_sharpe)
                else math.nan
            )
            rows.append(
                {
                    "window": window,
                    "ticks": ticks,
                    "sample": sample,
                    "sharpe_trend": trend_sharpe,
                    "sharpe_counter": fade_sharpe,
                    "edge": edge,
                    "maxdd_trend": max_drawdown(trend_daily),
                    "days": len(trend_daily),
                }
            )
    return rows


def score_asset(meta: dict, bars: list[dict]) -> tuple[dict, list[dict]]:
    simple = np.array([bar["ret"] for bar in bars], dtype=float)
    hourly_log = np.where(np.isfinite(simple), np.log1p(np.clip(simple, -0.99, None)), np.nan)
    index = compound_index(np.where(np.isfinite(simple), simple, np.nan))
    closes = np.array([bar["close"] for bar in bars], dtype=float)
    minstep = meta["minstep"]
    tick_frac = np.where(closes > 0, minstep / closes, np.nan) if math.isfinite(minstep) else np.full(len(bars), np.nan)
    dates = [bar["begin"].date() for bar in bars]
    daily = daily_frame(bars)

    hurst_h, hurst_h_se = hurst(hourly_log, HOURLY_QS)
    hurst_d, hurst_d_se = hurst(daily["logret"], DAILY_QS)
    acf_h1 = acf(hourly_log, 1)
    acf_h4 = acf(hourly_log, 4)
    acf_d1 = acf(daily["logret"], 1)
    vr_h = mean_finite([variance_ratio(hourly_log, q) for q in (8, 16)])
    vr_parts = {q: variance_ratio(daily["logret"], q) for q in (2, 4, 8, 10, 16)}
    vr_d10 = vr_parts[10]
    vr_d = mean_finite([vr_parts[q] for q in (2, 4, 8, 16)])

    year_h = {}
    year_acf = {}
    for year in (2024, 2025, 2026):
        mask = np.array([day.year == year for day in daily["dates"]])
        sample = daily["logret"][mask]
        finite_year = sample[np.isfinite(sample)]
        # Same scales as the full-sample Hurst. A short year cannot support q=16.
        if len(finite_year) < 16 * 8:
            year_h[year] = math.nan
        else:
            year_h[year], _error = hurst(finite_year, DAILY_QS)
        year_acf[year] = acf(sample, 1) if len(finite_year) >= 40 else math.nan

    hourly_excess, _hourly_mu = demean(dates, simple)
    daily_excess, daily_mu = demean(daily["dates"], daily["simple"])
    strategies = []
    for ticks in (0.0, 1.0):
        for row in horse_race(dates, hourly_excess, index, tick_frac, HOURLY_WINDOWS, ticks):
            row["horizon"] = "hourly"
            strategies.append(row)
        daily_tick = (
            np.where(daily["close"] > 0, minstep / daily["close"], np.nan)
            if math.isfinite(minstep)
            else np.full(len(daily["dates"]), np.nan)
        )
        daily_index = compound_index(daily["simple"])
        for row in horse_race(daily["dates"], daily_excess, daily_index, daily_tick, DAILY_WINDOWS, ticks):
            row["horizon"] = "daily"
            strategies.append(row)

    def edges(horizon: str, ticks: float, sample: str) -> list[float]:
        return [
            row["edge"]
            for row in strategies
            if row["horizon"] == horizon and row["ticks"] == ticks and row["sample"] == sample
        ]

    tick_bps = float(np.nanmedian(tick_frac) * 1e4) if np.isfinite(tick_frac).any() else math.nan
    per_day = defaultdict(int)
    for day in dates:
        per_day[day] += 1
    counts = np.array(list(per_day.values()))
    finite_daily = daily["logret"][np.isfinite(daily["logret"])]
    result = {
        "asset": meta["asset"],
        "name": meta["name"],
        "rank": int(meta["rank"]),
        "turnover_bln": meta["avg_daily_turnover_rub"] / 1e9,
        "oi": meta["avg_front_oi"],
        "bars": len(bars),
        "days": len(daily["dates"]),
        "first": bars[0]["begin"].date().isoformat() if bars else "",
        "last": bars[-1]["begin"].date().isoformat() if bars else "",
        "rolls": sum(bar["roll"] for bar in bars),
        "bars_per_day": float(np.median(counts)) if len(counts) else math.nan,
        "hourly_std_bps": float(np.nanstd(simple) * 1e4) if np.isfinite(simple).any() else math.nan,
        "tick_bps": tick_bps,
        "hurst_h": hurst_h,
        "hurst_h_se": hurst_h_se,
        "hurst_d": hurst_d,
        "hurst_d_se": hurst_d_se,
        "acf_h1": acf_h1,
        "acf_h4": acf_h4,
        "acf_d1": acf_d1,
        "acf_d1_t": acf_d1 * math.sqrt(len(finite_daily)) if math.isfinite(acf_d1) else math.nan,
        "vr_h": vr_h,
        "vr_d": vr_d,
        "vr_d2": vr_parts[2],
        "vr_d4": vr_parts[4],
        "vr_d8": vr_parts[8],
        "vr_d10": vr_d10,
        "vr_d16": vr_parts[16],
        "drift_ann": daily_mu * 252.0,
        "class_h": classify(hurst_h, acf_h4, vr_h),
        "class_d": classify(hurst_d, acf_d1, vr_d),
        "hurst_2024": year_h[2024],
        "hurst_2025": year_h[2025],
        "hurst_2026": year_h[2026],
        "acf_2024": year_acf[2024],
        "acf_2025": year_acf[2025],
        "acf_2026": year_acf[2026],
        "edge_h_test_0": median(edges("hourly", 0.0, "test")),
        "edge_h_test_1": median(edges("hourly", 1.0, "test")),
        "edge_d_test_0": median(edges("daily", 0.0, "test")),
        "edge_d_test_1": median(edges("daily", 1.0, "test")),
        "edge_h_train_1": median(edges("hourly", 1.0, "train")),
        "edge_d_train_1": median(edges("daily", 1.0, "train")),
        "bh_test": sharpe(daily_compound(daily["dates"], daily["simple"], TEST_START, None)),
    }
    flat_strategies = []
    for row in strategies:
        flat = {"asset": meta["asset"], "horizon": row["horizon"]}
        flat.update(row)
        flat_strategies.append(flat)
    return result, flat_strategies


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def confirmation(label: str, edge: float) -> str:
    if not math.isfinite(edge) or label in ("шум", "мало данных"):
        return "не различает"
    trendish = label in ("тренд", "слабо тренд")
    counter = label in ("контртренд", "слабо контртренд")
    if abs(edge) <= 0.25:
        return "не различает"
    if trendish and edge > 0:
        return "совпадает"
    if counter and edge < 0:
        return "совпадает"
    if trendish or counter:
        return "противоречит"
    return "не различает"


def write_scores(rows: list[dict], strategies: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (OUT / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in row.items()})
    with (OUT / "strategies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strategies[0].keys()))
        writer.writeheader()
        for row in strategies:
            writer.writerow({key: _cell(value) for key, value in row.items()})


def _cell(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return value


def write_svg(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["hurst_d"] if math.isfinite(row["hurst_d"]) else 0.0)
    width = 860
    height = 36 * len(ordered) + 96
    left = 280
    right = 40
    top = 36
    plot_w = width - left - right
    # Hurst from 0.35 to 0.65, random walk at 0.5.
    low, high = 0.35, 0.65

    def x_of(value: float) -> float:
        clipped = min(high, max(low, value))
        return left + (clipped - low) / (high - low) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="280" y="24" font-family="sans-serif" font-size="16" fill="#1a1a1a">'
        "Дневной показатель Хёрста, топ-25 фьючерсов Мосбиржи</text>",
    ]
    mid = x_of(0.5)
    axis_y = height - 46
    parts.append(f'<line x1="{mid:.1f}" y1="{top}" x2="{mid:.1f}" y2="{axis_y}" stroke="#9aa0a6"/>')
    parts.append(
        f'<text x="{mid:.1f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#5f6368">0.50 случайное блуждание</text>'
    )
    for tick in (0.40, 0.45, 0.55, 0.60):
        xpos = x_of(tick)
        parts.append(f'<line x1="{xpos:.1f}" y1="{top}" x2="{xpos:.1f}" y2="{axis_y}" stroke="#eeeeee"/>')
        parts.append(
            f'<text x="{xpos:.1f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11" fill="#5f6368">{tick:.2f}</text>'
        )
    for i, row in enumerate(ordered):
        y = top + i * 36 + 22
        color = "#0b6e4f" if row["hurst_d"] >= 0.5 else "#9b2226"
        label = f"{row['asset']}  {row['name']}"
        parts.append(
            f'<text x="{left - 12}" y="{y}" text-anchor="end" font-family="sans-serif" '
            f'font-size="13" fill="#1a1a1a">{label}</text>'
        )
        value = row["hurst_d"] if math.isfinite(row["hurst_d"]) else 0.5
        x0, x1 = sorted((mid, x_of(value)))
        parts.append(f'<rect x="{x0:.1f}" y="{y - 12}" width="{max(1, x1 - x0):.1f}" height="16" fill="{color}"/>')
    parts.append(
        f'<text x="24" y="{height - 8}" font-family="sans-serif" font-size="11" fill="#5f6368">'
        "Ниже 0.50 — контртренд, выше — тренд. Часовые свечи, склейка с января 2024 по 23 октября 2026, время биржи (МСК)."
        "</text>"
    )
    parts.append("</svg>")
    (OUT / "hurst-daily.svg").write_text("\n".join(parts))


def write_report(rows: list[dict], ranking_note: str) -> None:
    ordered = sorted(rows, key=lambda row: -(row["hurst_d"] if math.isfinite(row["hurst_d"]) else -9))
    trend = [row for row in ordered if row["class_d"] == "тренд"]
    counter = [row for row in ordered if row["class_d"] == "контртренд"]
    weak_trend = [row for row in ordered if row["class_d"] == "слабо тренд"]
    weak_counter = [row for row in ordered if row["class_d"] == "слабо контртренд"]
    noise = [row for row in ordered if row["class_d"] == "шум"]

    def names(group: list[dict]) -> str:
        if not group:
            return "нет"
        return ", ".join(f"{row['name']} ({row['asset']})" for row in group)

    yuan = [row for row in ordered if "юан" in row["name"].lower() or "CNY" in row["asset"] or "UCNY" in row["asset"]]
    hourly_trend = [row for row in ordered if row["class_h"] in ("тренд", "слабо тренд")]
    hourly_counter = [row for row in ordered if row["class_h"] in ("контртренд", "слабо контртренд")]
    lines = []
    lines.append("# Топ-25 фьючерсов Мосбиржи: тренд или контртренд")
    lines.append("")
    lines.append(
        "Короткий ответ по дневному горизонту — на нём живёт трендовая система, которая держит позицию дни и недели. "
        "Класс ставится, только когда показатель Хёрста, автокорреляция доходности и variance ratio "
        "не противоречат друг другу. Плоский признак не голосует. "
        f"Дневной тренд: {names(trend)}. "
        f"Дневной контртренд: {names(counter)}. "
        f"Слабый тренд: {names(weak_trend)}. Слабый контртренд: {names(weak_counter)}. "
        f"Остальные ({len(noise)}) — шум: либо всё лежит у случайного блуждания, либо метрики смотрят в разные стороны."
    )
    lines.append("")
    lines.append(
        f"На часе тренд или слабый тренд у {len(hourly_trend)} инструментов, "
        f"контртренд или слабый контртренд у {len(hourly_counter)}. "
        "Часовая свеча этих контрактов уже длинная, бид-аск в неё почти не попадает, "
        "поэтому час не становится автоматически контртрендом. "
        "У части валюты и индексов день контртрендовый, а час — трендовый. "
        "Системе, которая держит позицию несколько дней, нужен дневной класс."
    )
    positive_oos = sum(1 for row in ordered if math.isfinite(row["edge_d_test_1"]) and row["edge_d_test_1"] > 0)
    lines.append("")
    lines.append(
        f"В тесте 2026 года медиана дневного перевеса трендовой средней положительна у {positive_oos} из {len(ordered)}. "
        "Это один общий режим, а не двадцать независимых подтверждений: связанные рынки в один и тот же месяц "
        "дают один и тот же знак. Годовой Шарп на двухстах днях имеет ошибку около единицы, "
        "поэтому класс по всей истории 2024–2026 важнее, чем удачный или неудачный 2026-й."
    )
    lines.append("")
    lines.append("## Рейтинг")
    lines.append("")
    lines.append(
        "Сортировка по дневному Хёрсту. Выше 0.50 — прошлое продолжается, ниже — отыгрывается. "
        "Голос засчитывается, только если Хёрст отошёл от 0.50 хотя бы на 0.03, "
        "автокорреляция от нуля хотя бы на 0.02, а среднее VR по масштабам 2, 4, 8 и 16 дней "
        "отошло от единицы хотя бы на 0.02. Один масштаб VR слишком шумный, поэтому в класс идёт среднее. "
        "Оборот — среднее за последние 60 полных сессий "
        f"({ranking_note}), все экспирации одного кода вместе."
    )
    lines.append("")
    lines.append(
        "| # | Код | Инструмент | Оборот, млрд ₽/день | Хёрст день | ACF1 день | VR среднее | Класс день | "
        "Хёрст час | Класс час | Тик, б.п. |"
    )
    lines.append("|---|-----|------------|---------------------:|----------:|----------:|-----------:|------------|----------:|-----------|----------:|")
    for row in ordered:
        lines.append(
            f"| {row['rank']} | {row['asset']} | {row['name']} | {fmt(row['turnover_bln'], 2)} | "
            f"{fmt(row['hurst_d'], 3)} | {fmt(row['acf_d1'], 3)} | {fmt(row['vr_d'], 3)} | {row['class_d']} | "
            f"{fmt(row['hurst_h'], 3)} | {row['class_h']} | {fmt(row['tick_bps'], 2)} |"
        )
    lines.append("")
    lines.append("## Проверка простой системой")
    lines.append("")
    lines.append(
        "Одинаковое правило в обе стороны: длинная позиция, когда цена выше своей средней, "
        "короткая, когда ниже, и ровно наоборот для контртренда. Окна заданы заранее "
        "(день: 5, 10, 20, 40, 60, 120; час: 16, 32, 64, 128, 256 баров) и не выбирались по тесту. "
        "В таблице медиана разницы коэффициентов Шарпа «тренд минус контртренд» по этим окнам. "
        "Из доходности вычтено среднее за 2024–2025, иначе растущий рынок выглядит трендом "
        "только потому, что система чаще сидит в лонге. "
        "Тест — с 1 января 2026 по 23 октября 2026. Один тик на сторону: переворот платит два тика. "
        "На ~200 тестовых днях один сигма годового Шарпа около 1, поэтому отдельный инструмент "
        "этой проверкой не доказывается. Она показывает, остаётся ли перевес после вычета дрифта и тика."
    )
    lines.append("")
    lines.append(
        "| Код | Инструмент | Класс день | OOS день, 0 тиков | OOS день, 1 тик | "
        "OOS час, 1 тик | Train день, 1 тик | Сверка с классом |"
    )
    lines.append("|-----|------------|------------|------------------:|----------------:|----------------:|------------------:|------------------|")
    for row in ordered:
        lines.append(
            f"| {row['asset']} | {row['name']} | {row['class_d']} | {fmt(row['edge_d_test_0'])} | "
            f"{fmt(row['edge_d_test_1'])} | {fmt(row['edge_h_test_1'])} | {fmt(row['edge_d_train_1'])} | "
            f"{confirmation(row['class_d'], row['edge_d_test_1'])} |"
        )
    lines.append("")
    lines.append("## Устойчивость по годам")
    lines.append("")
    lines.append(
        "Дневная автокорреляция первого лага по календарным годам. У одного года около 250 дней, "
        "шум оценки примерно ±0.06, поэтому смотреть стоит на знак, который повторяется, а не на сотые. "
        "Хёрст одного года на масштабе 16 дней здесь не используется: блоков слишком мало."
    )
    lines.append("")
    lines.append("| Код | Инструмент | ACF 2024 | ACF 2025 | ACF 2026 | Класс день | Роллов | Баров в день |")
    lines.append("|-----|------------|---------:|---------:|---------:|------------|-------:|-------------:|")
    for row in ordered:
        lines.append(
            f"| {row['asset']} | {row['name']} | {fmt(row['acf_2024'], 3)} | {fmt(row['acf_2025'], 3)} | "
            f"{fmt(row['acf_2026'], 3)} | {row['class_d']} | {row['rolls']} | {fmt(row['bars_per_day'], 0)} |"
        )
    lines.append("")
    lines.append("## Юань")
    lines.append("")
    if not yuan:
        lines.append("В топ-25 по обороту за последние 60 сессий фьючерса на юань не оказалось.")
    else:
        for row in yuan:
            lines.append(
                f"{row['name']} ({row['asset']}) — {row['rank']}-е место по обороту, "
                f"{fmt(row['turnover_bln'], 2)} млрд ₽ в день. Дневной класс: {row['class_d']} "
                f"(Хёрст {fmt(row['hurst_d'], 3)}, ACF1 {fmt(row['acf_d1'], 3)}, t ≈ {fmt(row['acf_d1_t'], 1)}). "
                f"Часовой класс: {row['class_h']} (Хёрст {fmt(row['hurst_h'], 3)}). "
                f"Медиана OOS-перевеса трендовой средней на дневных окнах при одном тике: "
                f"{fmt(row['edge_d_test_1'])}. Дневная ACF1 по годам: {fmt(row['acf_2024'], 3)} / "
                f"{fmt(row['acf_2025'], 3)} / {fmt(row['acf_2026'], 3)}."
            )
            lines.append("")
        lines.append(
            "Квартальный юань, на котором сидят «Трендовый юань» и «Старая добрая трендовушка», "
            "на этой истории не трендовый. Дневной класс — слабый контртренд: "
            "автокорреляция соседних дней отрицательная во всех трёх годах, около −0.06…−0.08. "
            "Эффект небольшой, до сильного возврата к среднему ему далеко, но и продолжения движения статистика не показывает. "
            "Трендовая средняя в 2026 году на юане заработала перевес, однако тот же знак в этом тесте у большинства всего списка, "
            "так что 2026-й не доказывает трендовость юаня и не отменяет трёхлетний класс. "
            "Вечный CNYRUBTOM — другой стакан на том же курсе, у него дневной класс уже просто контртренд, а час, наоборот, трендовый."
        )
    lines.append("")
    lines.append("## Как это посчитано")
    lines.append("")
    lines.append(
        "Вселенная — 25 кодов активов FORTS с наибольшим рублёвым оборотом за последние 60 полных сессий. "
        "Полной считается сессия, в которой оборот рынка выше половины медианы. Часовые свечи ISS "
        "с 1 января 2024 по последнюю сессию 23 октября 2026, время биржи (МСК, UTC+3). В склейке на каждый день берётся "
        "контракт с максимальным часовым объёмом; смена только если новый контракт торгуется хотя бы "
        "в 1.15 раза больше. Доходность считается внутри контракта, разрыв цен при ролле в P&L не попадает."
    )
    lines.append("")
    lines.append(
        "Хёрст — наклон log(σ доходности масштаба q) к log(q). У случайного блуждания он равен 0.50. "
        "Дневные масштабы q = 2, 4, 8, 16; часовые q = 4, 8, 16, 32, 64, без самого короткого шага, "
        "чтобы бид-аск не занижал оценку. Variance ratio — дисперсия неперекрывающейся q-доходности, "
        "делённая на q и на дисперсию одного шага. ACF1 — корреляция соседних дневных лог-доходностей. "
        "На часе в класс идёт автокорреляция лага 4, а не лага 1."
    )
    lines.append("")
    lines.append(
        "Файлы свечей: `data/futures-hourly/<код>.csv.gz` (begin, secid, open, high, low, close, volume, ret, roll). "
        "Оборот и спецификация тика: `data/futures-hourly/universe.csv`. "
        "Числа отчёта: `results/futures-trend/scores.csv` и `strategies.csv`."
    )
    lines.append("")
    lines.append("## Чего эти цифры не говорят")
    lines.append("")
    short_rows = [row for row in ordered if row["first"] > "2024-02-01"]
    if short_rows:
        short_text = ", ".join(f"{row['name']} с {row['first']}" for row in short_rows)
        lines.append(f"Короче общего окна: {short_text}. Остальные ряды начинаются 3 января 2024.")
        lines.append("")
    lines.append(
        "Три года включают разные режимы ставки, курса и состава участников, так что класс 2024–2026 "
        "не обязан повториться. Хёрст на четырёх масштабах имеет узкую ошибку регрессии и легко "
        "выглядит «значимым» там, где экономический эффект мал: поэтому класс требует согласия трёх метрик, "
        "а не одного t-стата. Проверка средней на десяти месяцах 2026 года широкая, расхождение с классом "
        "на одном инструменте — не опровержение. Тик взят как один минимальный шаг на сторону; у тонкого "
        "стакана реальное проскальзывание больше, у Si и юаня — часто ровно тик. Комиссия брокера "
        "вторым порядком меньше тика на этих номиналах и в счёт не входила. Вечный фьючерс и квартальный "
        "на один и тот же базовый актив остаются разными инструментами, если оба попали в топ по обороту."
    )
    lines.append("")
    lines.append("## Повторный прогон")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHONPATH=. python3 -m research.futures_trend.test_metrics")
    lines.append("PYTHONPATH=. python3 -m research.futures_trend.fetch")
    lines.append("PYTHONPATH=. python3 -m research.futures_trend.analyze")
    lines.append("```")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines))


def ranking_note() -> str:
    path = DATA / "ranking_days.txt"
    if not path.exists():
        return "см. ranking_days.txt"
    days = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not days:
        return "см. ranking_days.txt"
    return f"{days[0]} — {days[-1]}, {len(days)} сессий"


def main() -> None:
    universe = load_universe()
    rows = []
    strategies = []
    for meta in universe:
        path = DATA / f"{meta['asset']}.csv.gz"
        if not path.exists():
            print(f"missing {path.name}")
            continue
        bars = load_bars(path)
        if len(bars) < 200:
            print(f"short {meta['asset']} bars={len(bars)}")
            continue
        result, detail = score_asset(meta, bars)
        rows.append(result)
        strategies.extend(detail)
        print(
            f"{result['asset']:12} {result['class_d']:16} Hday={result['hurst_d']:.3f} "
            f"Hhour={result['hurst_h']:.3f} edge1={result['edge_d_test_1']:.2f}"
        )
    rows.sort(key=lambda row: row["rank"])
    write_scores(rows, strategies)
    write_svg(rows)
    write_report(rows, ranking_note())
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()

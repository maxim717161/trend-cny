"""Trend versus mean-reversion statistics and a symmetric SMA horse-race."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

HOURLY_WINDOWS = (16, 32, 64, 128, 256)
DAILY_WINDOWS = (5, 10, 20, 40, 60, 120)
HOURLY_QS = (4, 8, 16, 32, 64)
DAILY_QS = (2, 4, 8, 16)


def acf(values: np.ndarray, lag: int) -> float:
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) <= lag + 5:
        return math.nan
    left = series[lag:]
    right = series[:-lag]
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float((left * left).sum() * (right * right).sum()))
    if denom == 0.0:
        return math.nan
    return float((left * right).sum() / denom)


def variance_ratio(logret: np.ndarray, q: int) -> float:
    series = np.asarray(logret, dtype=float)
    series = series[np.isfinite(series)]
    blocks = len(series) // q
    if blocks < 8:
        return math.nan
    chunk = series[: blocks * q]
    base = float(np.var(chunk, ddof=1))
    if base <= 0.0:
        return math.nan
    aggregated = chunk.reshape(blocks, q).sum(axis=1)
    return float(np.var(aggregated, ddof=1) / (q * base))


def hurst(logret: np.ndarray, qs: tuple[int, ...]) -> tuple[float, float]:
    """Slope of log(std of q-bar returns) on log(q). A random walk is 0.5."""
    series = np.asarray(logret, dtype=float)
    series = series[np.isfinite(series)]
    xs: list[float] = []
    ys: list[float] = []
    for q in qs:
        blocks = len(series) // q
        if blocks < 8:
            continue
        aggregated = series[: blocks * q].reshape(blocks, q).sum(axis=1)
        scale = float(np.std(aggregated, ddof=1))
        if scale <= 0.0:
            continue
        xs.append(math.log(q))
        ys.append(math.log(scale))
    if len(xs) < 3:
        return math.nan, math.nan
    coef, cov = np.polyfit(np.array(xs), np.array(ys), 1, cov=True)
    error = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else math.nan
    return float(coef[0]), error


def compound_index(simple: np.ndarray, start: float = 100.0) -> np.ndarray:
    index = np.empty(len(simple), dtype=float)
    level = start
    for i, ret in enumerate(simple):
        if np.isfinite(ret):
            level *= 1.0 + float(ret)
        index[i] = level
    return index


def sma_position(index: np.ndarray, window: int) -> np.ndarray:
    position = np.zeros(len(index), dtype=float)
    if window < 2 or len(index) < window:
        return position
    cumulative = np.cumsum(index)
    total = cumulative[window - 1 :] - np.concatenate(([0.0], cumulative[:-window]))
    average = total / window
    price = index[window - 1 :]
    sign = np.sign(price - average)
    position[window - 1 :] = sign
    return position


def strategy_returns(
    market: np.ndarray,
    position: np.ndarray,
    tick_frac: np.ndarray,
    ticks: float,
) -> np.ndarray:
    """Position decided on bar t earns market[t + 1]. A flip costs 2 ticks."""
    out = np.full(len(market), np.nan)
    held = 0.0
    for i in range(len(market) - 1):
        target = float(position[i])
        if not np.isfinite(target):
            target = 0.0
        turnover = abs(target - held)
        cost = turnover * float(tick_frac[i]) * ticks if np.isfinite(tick_frac[i]) else 0.0
        ret = market[i + 1]
        if np.isfinite(ret):
            out[i + 1] = target * float(ret) - cost
            held = target
    return out


def daily_compound(dates: list, returns: np.ndarray, date_from=None, date_to=None) -> np.ndarray:
    grouped: dict = defaultdict(list)
    for day, ret in zip(dates, returns):
        if date_from is not None and day < date_from:
            continue
        if date_to is not None and day >= date_to:
            continue
        if np.isfinite(ret):
            grouped[day].append(float(ret))
    daily = []
    for day in sorted(grouped):
        level = 1.0
        for ret in grouped[day]:
            level *= 1.0 + ret
        daily.append(level - 1.0)
    return np.array(daily, dtype=float)


def sharpe(daily: np.ndarray) -> float:
    if len(daily) < 40:
        return math.nan
    scale = float(np.std(daily, ddof=1))
    if scale <= 0.0:
        return math.nan
    return float(np.mean(daily) / scale * math.sqrt(252.0))


def max_drawdown(daily: np.ndarray) -> float:
    if len(daily) == 0:
        return math.nan
    equity = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def finite(values: list[float]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def median(values: list[float]) -> float:
    clean = finite(values)
    if not clean:
        return math.nan
    return float(np.median(np.array(clean)))

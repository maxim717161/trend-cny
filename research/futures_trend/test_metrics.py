"""Synthetic checks for the Hurst sign and the roll-safe return."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from research.futures_trend.metrics import acf, hurst, sharpe, sma_position, strategy_returns, variance_ratio
from research.futures_trend.series import build_continuous


def _ar(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, 1.0, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + shock[i]
    return out


def test_hurst_sign():
    qs = (2, 4, 8, 16, 32, 64)
    trend_h, _ = hurst(_ar(0.6, 8000, 1), qs)
    revert_h, _ = hurst(_ar(-0.6, 8000, 1), qs)
    noise_h, _ = hurst(_ar(0.0, 8000, 1), qs)
    assert trend_h > 0.55, trend_h
    assert revert_h < 0.45, revert_h
    assert 0.45 < noise_h < 0.55, noise_h
    assert acf(_ar(0.6, 4000, 2), 1) > 0.3
    assert acf(_ar(-0.6, 4000, 2), 1) < -0.3
    ratio = variance_ratio(_ar(0.0, 8000, 3), 8)
    assert 0.85 < ratio < 1.15, ratio


def test_roll_does_not_book_the_gap():
    start = datetime(2024, 1, 2, 10, 0, 0)
    records = []
    # Contract A trades day 1 at ~100, contract B trades both days at ~200.
    # Volume hands the book to B on day 2. The price level doubles, the return must not.
    for hour in range(4):
        moment = start + timedelta(hours=hour)
        records.append(_bar("A", moment, 100 + hour, volume=100))
        records.append(_bar("B", moment, 200 + hour, volume=10))
    day2 = start + timedelta(days=1)
    for hour in range(4):
        moment = day2 + timedelta(hours=hour)
        records.append(_bar("A", moment, 110 + hour, volume=10))
        records.append(_bar("B", moment, 204 + hour, volume=100))
    stitched = build_continuous(records, hysteresis=1.15)
    assert [bar["secid"] for bar in stitched[:4]] == ["A"] * 4
    assert [bar["secid"] for bar in stitched[4:]] == ["B"] * 4
    assert stitched[4]["roll"] == 1
    # B's own move from 203 to 204, not the jump from A's 103 to B's 204.
    assert abs(stitched[4]["ret"] - (204 / 203 - 1)) < 1e-12
    assert abs(stitched[1]["ret"] - (101 / 100 - 1)) < 1e-12


def test_sma_prefers_the_right_side_of_persistence():
    rets = _ar(0.5, 1500, 4) * 0.01
    index = np.cumprod(1.0 + rets) * 100
    # The strategy earns the next return. Persistence is in `rets` itself,
    # so trade the return series directly with an SMA of the price index.
    position = sma_position(index, 20)
    tick = np.zeros(len(rets))
    trend = strategy_returns(rets, position, tick, ticks=0.0)
    fade = strategy_returns(rets, -position, tick, ticks=0.0)
    dates = [i // 1 for i in range(len(rets))]
    # One "day" per bar keeps the helper applicable via a flat date list.
    from research.futures_trend.metrics import daily_compound

    trend_sharpe = sharpe(daily_compound(dates, trend))
    fade_sharpe = sharpe(daily_compound(dates, fade))
    assert trend_sharpe > fade_sharpe
    assert trend_sharpe > 0


def _bar(secid: str, moment: datetime, close: float, volume: float) -> dict:
    return {
        "secid": secid,
        "begin": moment,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
    }


if __name__ == "__main__":
    test_hurst_sign()
    test_roll_does_not_book_the_gap()
    test_sma_prefers_the_right_side_of_persistence()
    print("ok")

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.filters.regime_filter import RegimeFilter


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=260, freq="B", tz="UTC")
    spy = np.linspace(100, 130, len(idx))
    qqq = np.linspace(90, 120, len(idx))
    return pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=idx)


def test_get_regime_series_shape() -> None:
    rf = RegimeFilter(benchmark="SPY", ma_window=50)
    regime = rf.get_regime_series(_prices())
    assert len(regime) == len(_prices())
    assert regime.dtype == bool


def test_is_bullish_returns_boolean() -> None:
    prices = _prices()
    rf = RegimeFilter(benchmark="SPY", ma_window=50)
    val = rf.is_bullish(prices, prices.index[-1])
    assert isinstance(val, bool)


def test_filter_signals_zeroes_bearish_periods() -> None:
    prices = _prices()
    rf = RegimeFilter(benchmark="SPY", ma_window=50)
    regime = rf.get_regime_series(prices)
    sig_idx = pd.MultiIndex.from_product([prices.index, ["QQQ"]], names=["timestamp", "symbol"])
    signals = pd.Series(1, index=sig_idx)
    filtered = rf.filter_signals(signals, regime)
    first_ts = prices.index[0]
    assert filtered.loc[(first_ts, "QQQ")] == 0

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.metrics import cagr, max_drawdown, profit_factor, sharpe_ratio, summary, win_rate


def test_sharpe_positive_series() -> None:
    returns = pd.Series([0.01, 0.005, 0.012, -0.002, 0.004])
    assert sharpe_ratio(returns) != 0.0


def test_max_drawdown() -> None:
    equity = pd.Series([100, 110, 105, 120, 90, 130], index=pd.date_range("2024-01-01", periods=6, freq="D"))
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_cagr_and_summary() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="Y")
    equity = pd.Series([100, 110, 121, 133.1, 146.41, 161.051], index=idx)
    trades = pd.Series([10, -5, 15, -3], dtype=float)
    assert cagr(equity) > 0
    s = summary(equity, trades)
    assert "sharpe_ratio" in s
    assert s["profit_factor"] == pytest.approx(profit_factor(trades))
    assert s["win_rate"] == pytest.approx(win_rate(trades))

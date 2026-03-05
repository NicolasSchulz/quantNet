from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.cost_model import CostModel
from backtesting.optimizer import LookbackOptimizer
from strategies.examples.simple_momentum import SimpleMomentumStrategy


def _build_data() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2020-01-01", periods=900, freq="B", tz="UTC")
    symbols = ["SPY", "XLK", "XLF", "EEM", "TLT", "GLD"]
    data: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols):
        raw_base = 100 + np.cumsum(np.random.default_rng(i).normal(0.05 * (i - 2), 1.0, len(idx)))
        base = np.maximum(raw_base, 5.0)
        open_px = np.maximum(base * (1 + np.random.default_rng(i + 100).normal(0.0, 0.001, len(idx))), 4.0)
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": open_px,
                "high": open_px * 1.01,
                "low": open_px * 0.99,
                "close": base,
                "volume": 1000,
                "symbol": symbol,
                "asset_class": "etf",
            },
            index=idx,
        )
        df.index.name = "timestamp"
        data[symbol] = df
    return data


def test_optimize_single_param_returns_result() -> None:
    optimizer = LookbackOptimizer(
        strategy_class=SimpleMomentumStrategy,
        data=_build_data(),
        cost_model=CostModel(),
        initial_capital=100000,
    )
    result = optimizer.optimize("formation_months", [3, 6])
    assert isinstance(result.best_params, dict)
    assert "params" in result.results_df.columns


def test_optimize_grid_has_expected_rows() -> None:
    optimizer = LookbackOptimizer(
        strategy_class=SimpleMomentumStrategy,
        data=_build_data(),
        cost_model=CostModel(),
        initial_capital=100000,
    )
    grid = {
        "formation_months": [3, 6],
        "skip_months": [0, 1],
        "rebalance_freq": ["M"],
        "use_regime_filter": [False],
    }
    out = optimizer.optimize_grid(grid)
    assert len(out) == 4
    assert {"params", "sharpe", "cagr", "max_drawdown", "calmar", "n_trades"}.issubset(out.columns)


def test_overfit_warning_column_present() -> None:
    optimizer = LookbackOptimizer(
        strategy_class=SimpleMomentumStrategy,
        data=_build_data(),
        cost_model=CostModel(),
        initial_capital=100000,
    )
    out = optimizer.optimize_grid(
        {
            "formation_months": [12],
            "skip_months": [1],
            "rebalance_freq": ["M"],
            "use_regime_filter": [False],
        }
    )
    assert "overfit_warning" in out.columns

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from backtesting.cost_model import CostModel
from backtesting.engine import run


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_sharpe: float
    results_df: pd.DataFrame


class LookbackOptimizer:
    def __init__(
        self,
        strategy_class: type,
        data: dict[str, pd.DataFrame],
        cost_model: CostModel,
        initial_capital: float,
        train_fraction: float = 0.70,
        overfit_ratio_warning: float = 1.5,
        sizing_method: str = "equal_weight",
    ) -> None:
        self.strategy_class = strategy_class
        self.data = data
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.train_fraction = train_fraction
        self.overfit_ratio_warning = overfit_ratio_warning
        self.sizing_method = sizing_method

    def _aligned_index(self) -> pd.DatetimeIndex:
        return pd.concat(
            {sym: df["close"].astype(float) for sym, df in self.data.items()}, axis=1
        ).dropna(how="any").index

    def _split_data(self) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        idx = self._aligned_index()
        if len(idx) < 20:
            raise ValueError("Not enough aligned data for walk-forward optimization.")

        split_loc = int(len(idx) * self.train_fraction)
        split_loc = max(5, min(split_loc, len(idx) - 5))
        split_date = idx[split_loc - 1]

        train = {sym: df[df.index <= split_date].copy() for sym, df in self.data.items()}
        test = {sym: df[df.index > split_date].copy() for sym, df in self.data.items()}
        return train, test

    def _evaluate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        train_data, test_data = self._split_data()

        strategy_train = self.strategy_class(**params)
        train_res = run(
            strategy=strategy_train,
            data=train_data,
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
            rebalance_frequency=params.get("rebalance_freq", "M"),
            sizing_method=self.sizing_method,
        )

        strategy_test = self.strategy_class(**params)
        test_res = run(
            strategy=strategy_test,
            data=test_data,
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
            rebalance_frequency=params.get("rebalance_freq", "M"),
            sizing_method=self.sizing_method,
        )

        train_sharpe = float(train_res.metrics.get("sharpe_ratio", 0.0))
        test_sharpe = float(test_res.metrics.get("sharpe_ratio", 0.0))

        overfit = test_sharpe != 0 and train_sharpe / max(test_sharpe, 1e-9) > self.overfit_ratio_warning
        if test_sharpe <= 0 and train_sharpe > 0:
            overfit = True

        return {
            "params": params,
            "train_sharpe": train_sharpe,
            "sharpe": test_sharpe,
            "cagr": float(test_res.metrics.get("cagr", 0.0)),
            "max_drawdown": float(test_res.metrics.get("max_drawdown", 0.0)),
            "calmar": float(test_res.metrics.get("calmar_ratio", 0.0)),
            "n_trades": int(len(test_res.trades)),
            "overfit_warning": bool(overfit),
        }

    def optimize(self, param_name: str, param_range: list[Any]) -> OptimizationResult:
        rows = []
        for value in param_range:
            rows.append(self._evaluate_params({param_name: value}))

        results_df = pd.DataFrame(rows).sort_values("train_sharpe", ascending=False).reset_index(drop=True)
        best_row = results_df.iloc[0]
        return OptimizationResult(
            best_params=best_row["params"],
            best_sharpe=float(best_row["sharpe"]),
            results_df=results_df,
        )

    def optimize_grid(self, param_grid: dict[str, list[Any]]) -> pd.DataFrame:
        keys = list(param_grid.keys())
        rows = []
        for values in product(*[param_grid[k] for k in keys]):
            params = dict(zip(keys, values))
            rows.append(self._evaluate_params(params))

        results_df = pd.DataFrame(rows)
        # Required column order plus additional diagnostic columns.
        ordered_cols = [
            "params",
            "sharpe",
            "cagr",
            "max_drawdown",
            "calmar",
            "n_trades",
            "train_sharpe",
            "overfit_warning",
        ]
        results_df = results_df[ordered_cols].sort_values("sharpe", ascending=False).reset_index(drop=True)
        return results_df

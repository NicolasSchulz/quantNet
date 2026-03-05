from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.factors.momentum import cross_sectional_momentum


@dataclass
class SimpleMomentumStrategy(BaseStrategy):
    """Cross-sectional 12-1 momentum strategy with periodic rebalancing."""

    lookback_months: int = 12
    skip_months: int = 1
    rebalance_frequency: str = "M"
    long_quantile: float = 0.2
    short_quantile: float = 0.2

    def get_name(self) -> str:
        return "SimpleMomentum"

    def get_parameters(self) -> dict[str, float | int | str]:
        return {
            "lookback_months": self.lookback_months,
            "skip_months": self.skip_months,
            "rebalance_frequency": self.rebalance_frequency,
            "long_quantile": self.long_quantile,
            "short_quantile": self.short_quantile,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate stacked (timestamp, symbol) signal series.

        Args:
            data: Price matrix (index=time, columns=symbols).
        """
        if data.empty:
            return pd.Series(dtype=float)

        prices = data.sort_index().copy()
        rebalance_dates = prices.resample(self.rebalance_frequency).last().index
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)

        for rebalance_date in rebalance_dates:
            hist = prices.loc[:rebalance_date]
            monthly = hist.resample("M").last().dropna(how="all")
            need = self.lookback_months + self.skip_months + 1
            if len(monthly) < need:
                continue

            if self.lookback_months == 12 and self.skip_months == 1:
                scores = cross_sectional_momentum(hist).dropna()
            else:
                end_idx = -1 - self.skip_months
                start_idx = end_idx - self.lookback_months
                lookback_prices = monthly.iloc[[start_idx, end_idx]]
                scores = (lookback_prices.iloc[1] / lookback_prices.iloc[0] - 1.0).dropna()
            scores = scores.dropna()
            if scores.empty:
                continue

            n_assets = len(scores)
            n_long = max(1, int(n_assets * self.long_quantile))
            n_short = max(1, int(n_assets * self.short_quantile))

            longs = scores.nlargest(n_long).index
            shorts = scores.nsmallest(n_short).index

            if rebalance_date in signals.index:
                signals.loc[rebalance_date, longs] = 1
                signals.loc[rebalance_date, shorts] = -1

        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype(int)
        stacked = signals.stack()
        stacked.index.names = ["timestamp", "symbol"]
        stacked.name = "signal"
        return stacked

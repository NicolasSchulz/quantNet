from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.factors.momentum import cross_sectional_momentum
from strategies.filters.regime_filter import RegimeFilter


@dataclass
class SimpleMomentumStrategy(BaseStrategy):
    """Cross-sectional 12-1 momentum strategy with periodic rebalancing."""

    formation_months: int = 12
    skip_months: int = 1
    rebalance_freq: str = "M"
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    use_regime_filter: bool = True
    regime_benchmark: str = "SPY"
    regime_ma_window: int = 200
    tradable_symbols: list[str] | None = None
    regime_series: pd.Series | None = None

    def get_name(self) -> str:
        return "SimpleMomentum"

    def get_parameters(self) -> dict[str, float | int | str]:
        return {
            "formation_months": self.formation_months,
            "skip_months": self.skip_months,
            "rebalance_freq": self.rebalance_freq,
            "long_quantile": self.long_quantile,
            "short_quantile": self.short_quantile,
            "use_regime_filter": self.use_regime_filter,
            "regime_benchmark": self.regime_benchmark,
            "regime_ma_window": self.regime_ma_window,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate stacked (timestamp, symbol) signal series.

        Args:
            data: Price matrix (index=time, columns=symbols).
        """
        if data.empty:
            return pd.Series(dtype=float)

        prices = data.sort_index().copy()
        rebalance_dates = prices.resample(self.rebalance_freq).last().index
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)
        if self.tradable_symbols is not None:
            eligible_symbols = [sym for sym in self.tradable_symbols if sym in prices.columns]
        else:
            eligible_symbols = list(prices.columns)

        for rebalance_date in rebalance_dates:
            hist = prices.loc[:rebalance_date, eligible_symbols]
            monthly = hist.resample("M").last().dropna(how="all")
            need = self.formation_months + self.skip_months + 1
            if len(monthly) < need:
                continue

            if self.formation_months == 12 and self.skip_months == 1:
                scores = cross_sectional_momentum(hist).dropna()
            else:
                end_idx = -1 - self.skip_months
                start_idx = end_idx - self.formation_months
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

        if self.use_regime_filter:
            regime_filter = RegimeFilter(
                benchmark=self.regime_benchmark,
                ma_window=self.regime_ma_window,
            )
            self.regime_series = regime_filter.get_regime_series(prices)
            # Long-only when regime filter is enabled: convert shorts to flat.
            signals = signals.clip(lower=0)
            stacked_pre = signals.stack()
            stacked_pre.index.names = ["timestamp", "symbol"]
            filtered = regime_filter.filter_signals(stacked_pre, self.regime_series)
            signals = filtered.unstack("symbol").reindex(signals.index).fillna(0).astype(int)
        else:
            self.regime_series = pd.Series(True, index=signals.index, name="regime_bullish")

        # Ensure non-tradable helper columns (e.g., benchmark) stay flat.
        for col in signals.columns:
            if col not in eligible_symbols:
                signals[col] = 0

        stacked = signals.stack()
        stacked.index.names = ["timestamp", "symbol"]
        stacked.name = "signal"
        return stacked

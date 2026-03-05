from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RegimeFilter:
    benchmark: str = "SPY"
    ma_window: int = 200

    def is_bullish(self, prices: pd.DataFrame, date: pd.Timestamp) -> bool:
        regime_series = self.get_regime_series(prices)
        date = pd.Timestamp(date)
        if date not in regime_series.index:
            return False
        return bool(regime_series.loc[date])

    def get_regime_series(self, prices: pd.DataFrame) -> pd.Series:
        benchmark = self.benchmark.upper()
        if benchmark not in prices.columns:
            raise ValueError(
                f"Benchmark '{benchmark}' not found in prices columns."
            )

        benchmark_close = prices[benchmark].astype(float)
        ma = benchmark_close.rolling(window=self.ma_window, min_periods=self.ma_window).mean()
        regime = benchmark_close > ma
        regime = regime.fillna(False)
        regime.name = "regime_bullish"
        return regime

    def filter_signals(self, signals: pd.Series, regime: pd.Series) -> pd.Series:
        if isinstance(signals.index, pd.MultiIndex):
            timestamps = signals.index.get_level_values("timestamp")
            aligned_regime = regime.reindex(timestamps).fillna(False)
            filtered = signals.copy().astype(float)
            filtered[~aligned_regime.to_numpy()] = 0.0
            return filtered.astype(int)

        aligned_regime = regime.reindex(signals.index).fillna(False)
        filtered = signals.copy().astype(float)
        filtered[~aligned_regime] = 0.0
        return filtered.astype(int)

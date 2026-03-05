from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_momentum(universe: pd.DataFrame) -> pd.Series:
    """Compute 12-1 momentum score per symbol from price matrix.

    Args:
        universe: Price matrix with datetime index and symbols as columns.

    Returns:
        Series indexed by symbol with momentum scores (higher is better).
    """
    if universe.empty:
        return pd.Series(dtype=float)

    monthly = universe.resample("M").last().dropna(how="all")
    if len(monthly) < 13:
        return pd.Series(index=universe.columns, data=np.nan, dtype=float)

    ret_12m = monthly.iloc[-1] / monthly.iloc[-13] - 1.0
    ret_1m = monthly.iloc[-1] / monthly.iloc[-2] - 1.0
    scores = ret_12m - ret_1m
    return scores.sort_values(ascending=False)


def time_series_momentum(prices: pd.Series, lookback: int) -> pd.Series:
    """Compute time-series momentum signal from rolling returns.

    Signal = sign(prices / prices.shift(lookback) - 1), mapped to {-1,0,1}.
    """
    if lookback <= 0:
        raise ValueError("lookback must be > 0")

    rolling_return = prices / prices.shift(lookback) - 1.0
    signal = np.sign(rolling_return).fillna(0.0)
    return signal.astype(int)

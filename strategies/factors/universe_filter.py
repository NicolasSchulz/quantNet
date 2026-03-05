from __future__ import annotations

import pandas as pd


def filter_by_liquidity(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_avg_dollar_volume: float,
    window: int = 20,
) -> list[str]:
    """Return symbols passing a minimum rolling average dollar-volume threshold."""
    if prices.empty or volumes.empty:
        return []
    if window <= 0:
        raise ValueError("window must be > 0")

    dollar_volume = prices * volumes
    avg_dv = dollar_volume.rolling(window).mean().iloc[-1].dropna()
    passed = avg_dv[avg_dv >= min_avg_dollar_volume]
    return sorted(passed.index.tolist())

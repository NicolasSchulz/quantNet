from __future__ import annotations

import pandas as pd


def equal_weight(n_positions: int, capital: float) -> float:
    if n_positions <= 0:
        return 0.0
    if capital < 0:
        raise ValueError("capital must be non-negative")
    return float(capital / n_positions)


def volatility_parity(
    signals: pd.Series,
    returns: pd.DataFrame,
    capital: float,
    target_vol: float = 0.10,
) -> pd.Series:
    if capital < 0:
        raise ValueError("capital must be non-negative")
    if signals.empty or returns.empty:
        return pd.Series(dtype=float)

    active = signals[signals != 0].index
    if len(active) == 0:
        return pd.Series(dtype=float)

    vol = returns[active].std(ddof=0) * (252 ** 0.5)
    inv_vol = 1.0 / vol.replace(0, pd.NA)
    inv_vol = inv_vol.dropna()
    if inv_vol.empty:
        return pd.Series(dtype=float)

    weights = inv_vol / inv_vol.sum()
    scaled_weights = weights * target_vol
    notional = scaled_weights * capital
    direction = signals.loc[scaled_weights.index].astype(float)
    return notional * direction


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    fraction: float = 0.25,
) -> float:
    if capital < 0:
        raise ValueError("capital must be non-negative")
    if avg_loss <= 0:
        return 0.0

    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    b = avg_win / avg_loss if avg_loss > 0 else 0.0
    if b <= 0:
        return 0.0

    kelly = p - (q / b)
    kelly = max(0.0, kelly)
    return float(capital * kelly * fraction)

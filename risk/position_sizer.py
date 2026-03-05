from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class VolatilityParitySizer:
    target_vol: float = 0.10
    lookback_days: int = 60
    max_position: float = 0.20

    def compute_weights(
        self,
        signals: pd.Series,
        returns: pd.DataFrame,
        date: pd.Timestamp,
    ) -> pd.Series:
        if self.lookback_days <= 1:
            raise ValueError("lookback_days must be > 1")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")

        active = signals[signals > 0].index
        if len(active) == 0:
            return pd.Series(0.0, index=signals.index, dtype=float)

        history = returns.loc[:date, active].tail(self.lookback_days)
        if history.empty:
            return pd.Series(0.0, index=signals.index, dtype=float)

        vol = history.std(ddof=0).replace(0, pd.NA).dropna()
        if vol.empty:
            return pd.Series(0.0, index=signals.index, dtype=float)

        inv_vol = 1.0 / vol
        base_weights = inv_vol / inv_vol.sum()

        portfolio_daily_vol = float((history[base_weights.index] @ base_weights).std(ddof=0))
        if portfolio_daily_vol > 0:
            scale = min(1.0, (self.target_vol / (252**0.5)) / portfolio_daily_vol)
        else:
            scale = 0.0

        weights = (base_weights * scale).clip(upper=self.max_position)
        total = float(weights.sum())
        if total > 1.0:
            weights = weights / total

        out = pd.Series(0.0, index=signals.index, dtype=float)
        out.loc[weights.index] = weights.values
        return out

    def compare_with_equal_weight(
        self,
        signals: pd.Series,
        returns: pd.DataFrame,
        date: pd.Timestamp,
    ) -> dict[str, pd.Series]:
        active = signals[signals > 0].index
        eq = pd.Series(0.0, index=signals.index, dtype=float)
        if len(active) > 0:
            eq.loc[active] = 1.0 / len(active)

        vp = self.compute_weights(signals=signals, returns=returns, date=date)
        return {"equal_weight": eq, "volatility_parity": vp}

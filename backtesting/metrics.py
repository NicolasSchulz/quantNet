from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    if returns is None or returns.empty:
        return 0.0
    clean = returns.dropna()
    if clean.empty or clean.std(ddof=0) == 0:
        return 0.0
    daily_rf = risk_free_rate / 252.0
    excess = clean - daily_rf
    return float(np.sqrt(252.0) * excess.mean() / clean.std(ddof=0))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def cagr(equity_curve: pd.Series) -> float:
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return float((end / start) ** (1 / years) - 1.0)


def calmar_ratio(equity_curve: pd.Series) -> float:
    dd = abs(max_drawdown(equity_curve))
    if dd == 0:
        return 0.0
    return float(cagr(equity_curve) / dd)


def win_rate(trades: pd.Series) -> float:
    if trades is None or trades.empty:
        return 0.0
    pnl = trades.dropna()
    if pnl.empty:
        return 0.0
    return float((pnl > 0).sum() / len(pnl))


def profit_factor(trades: pd.Series) -> float:
    if trades is None or trades.empty:
        return 0.0
    pnl = trades.dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss == 0:
        return 0.0 if gross_profit == 0 else float("inf")
    return gross_profit / gross_loss


def summary(equity_curve: pd.Series, trades: pd.Series) -> dict[str, float]:
    returns = equity_curve.pct_change().dropna() if equity_curve is not None else pd.Series(dtype=float)
    return {
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "cagr": cagr(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
    }

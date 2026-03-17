from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

try:
    from api.schemas.trades import Trade
except ImportError:
    from schemas.trades import Trade


class TradingStats(BaseModel):
    total_trades: int
    open_trades: int
    closed_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_win_pct: float
    max_loss_pct: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    cagr: float
    calmar_ratio: float
    total_pnl: float
    total_commission: float
    avg_holding_hours: float
    best_trade: Optional[Trade] = None
    worst_trade: Optional[Trade] = None


class EquityPoint(BaseModel):
    timestamp: datetime
    equity: float
    drawdown: float


class EquityCurveResponse(BaseModel):
    points: list[EquityPoint]
    benchmark: list[EquityPoint]


class PnlDistributionBucket(BaseModel):
    range: str
    count: int
    pct: float


class PnlDistributionResponse(BaseModel):
    buckets: list[PnlDistributionBucket]


class SymbolPerformanceRow(BaseModel):
    symbol: str
    trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    sharpe: float


class SymbolPerformanceResponse(BaseModel):
    rows: list[SymbolPerformanceRow]


class ExitReasonPerformanceRow(BaseModel):
    exit_reason: str
    trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float


class ExitReasonPerformanceResponse(BaseModel):
    rows: list[ExitReasonPerformanceRow]

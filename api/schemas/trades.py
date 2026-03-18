from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


Direction = Literal["LONG", "SHORT"]
TradeStatus = Literal["OPEN", "CLOSED", "CANCELLED"]
ExitReason = Literal["TAKE_PROFIT", "STOP_LOSS", "TIME_BARRIER", "MANUAL"]


class Trade(BaseModel):
    id: str
    symbol: str
    asset_class: Literal["equity", "crypto"]
    direction: Direction
    status: TradeStatus
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: float
    slippage: float
    strategy: str
    signal_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    exit_reason: Optional[ExitReason] = None
    notes: Optional[str] = None


class TradesResponse(BaseModel):
    trades: list[Trade]
    total: int
    page: int
    pages: int


class TradesSummary(BaseModel):
    open: int
    closed: int
    today_pnl: float
    total_pnl: float
    win_rate: float


class PriceBar(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class PriceContextResponse(BaseModel):
    bars: list[PriceBar]
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None

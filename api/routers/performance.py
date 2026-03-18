from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter

try:
    from api.schemas.performance import (
        EquityCurveResponse,
        ExitReasonPerformanceResponse,
        PnlDistributionResponse,
        SymbolPerformanceResponse,
        TradingStats,
    )
    from api.services.performance_service import (
        get_equity_curve,
        get_performance_by_exit_reason,
        get_performance_by_symbol,
        get_pnl_distribution,
        get_trading_stats,
    )
except ImportError:
    from schemas.performance import (
        EquityCurveResponse,
        ExitReasonPerformanceResponse,
        PnlDistributionResponse,
        SymbolPerformanceResponse,
        TradingStats,
    )
    from services.performance_service import (
        get_equity_curve,
        get_performance_by_exit_reason,
        get_performance_by_symbol,
        get_pnl_distribution,
        get_trading_stats,
    )

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("/stats", response_model=TradingStats)
def performance_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    asset_class: Optional[str] = None,
) -> dict[str, object]:
    return get_trading_stats(start_date=start_date, end_date=end_date, symbol=symbol, strategy=strategy, asset_class=asset_class)


@router.get("/equity-curve", response_model=EquityCurveResponse)
def performance_equity_curve(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    asset_class: Optional[str] = None,
) -> dict[str, object]:
    return get_equity_curve(start_date=start_date, end_date=end_date, asset_class=asset_class)


@router.get("/pnl-distribution", response_model=PnlDistributionResponse)
def performance_pnl_distribution(asset_class: Optional[str] = None) -> dict[str, object]:
    return get_pnl_distribution(asset_class=asset_class)


@router.get("/by-symbol", response_model=SymbolPerformanceResponse)
def performance_by_symbol(asset_class: Optional[str] = None) -> dict[str, object]:
    return get_performance_by_symbol(asset_class=asset_class)


@router.get("/by-exit-reason", response_model=ExitReasonPerformanceResponse)
def performance_by_exit_reason(asset_class: Optional[str] = None) -> dict[str, object]:
    return get_performance_by_exit_reason(asset_class=asset_class)

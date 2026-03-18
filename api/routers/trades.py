from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

try:
    from api.schemas.trades import PriceContextResponse, Trade, TradesResponse, TradesSummary
    from api.services.trade_service import (
        filter_trades,
        get_trade,
        get_trade_price_context,
        get_trade_summary,
        load_trades,
        paginate_trades,
    )
except ImportError:
    from schemas.trades import PriceContextResponse, Trade, TradesResponse, TradesSummary
    from services.trade_service import (
        filter_trades,
        get_trade,
        get_trade_price_context,
        get_trade_summary,
        load_trades,
        paginate_trades,
    )

router = APIRouter(prefix="/trades", tags=["trades"])


@router.get("", response_model=TradesResponse)
def list_trades(
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    status: Optional[str] = None,
    asset_class: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strategy: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
) -> dict[str, object]:
    trades = filter_trades(
        load_trades(),
        symbol=symbol,
        direction=direction,
        status=status,
        asset_class=asset_class,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy,
    )
    return paginate_trades(trades, page=page, page_size=page_size)


@router.get("/summary", response_model=TradesSummary)
def trades_summary(asset_class: Optional[str] = None) -> dict[str, float]:
    summary = get_trade_summary(filter_trades(load_trades(), asset_class=asset_class))
    return {
        "open": int(summary["open"]),
        "closed": int(summary["closed"]),
        "today_pnl": float(summary["today_pnl"]),
        "total_pnl": float(summary["total_pnl"]),
        "win_rate": float(summary["win_rate"]),
    }


@router.get("/price-context/{trade_id}", response_model=PriceContextResponse)
def trade_price_context(trade_id: str) -> dict[str, object]:
    context = get_trade_price_context(trade_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return context


@router.get("/{trade_id}", response_model=Trade)
def trade_detail(trade_id: str) -> dict[str, object]:
    trade = get_trade(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade.to_dict()

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

import pandas as pd


Direction = Literal["LONG", "SHORT"]
TradeStatus = Literal["OPEN", "CLOSED", "CANCELLED"]
ExitReason = Literal["TAKE_PROFIT", "STOP_LOSS", "TIME_BARRIER", "MANUAL"]


@dataclass(frozen=True)
class TradeRecord:
    id: str
    symbol: str
    direction: Direction
    status: TradeStatus
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    commission: float
    slippage: float
    strategy: str
    signal_confidence: Optional[float]
    exit_reason: Optional[ExitReason]
    notes: Optional[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def is_live_mode() -> bool:
    return os.getenv("LIVE_MODE", "false").lower() == "true"


@lru_cache(maxsize=1)
def _mock_trades() -> tuple[TradeRecord, ...]:
    rng = random.Random(42)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
    strategies = ["MLStrategy_SPY_v1", "MomentumSwing_v2", "MeanReversion_v1"]
    trades: list[TradeRecord] = []
    for idx in range(160):
        symbol = rng.choice(symbols)
        direction: Direction = rng.choice(["LONG", "SHORT"])
        entry_time = now - timedelta(hours=6 * idx + rng.randint(1, 5))
        is_open = idx < 6
        status: TradeStatus = "OPEN" if is_open else ("CANCELLED" if idx % 27 == 0 else "CLOSED")
        entry_price = round(rng.uniform(90, 540), 2)
        qty = round(rng.uniform(10, 150), 2)
        confidence = round(rng.uniform(0.35, 0.91), 2)
        commission = round(rng.uniform(0.2, 3.5), 2)
        slippage = round(rng.uniform(0.05, 1.2), 2)
        exit_time: Optional[datetime] = None
        exit_price: Optional[float] = None
        pnl: Optional[float] = None
        pnl_pct: Optional[float] = None
        exit_reason: Optional[ExitReason] = None
        if status == "CLOSED":
            hold_hours = rng.randint(2, 72)
            exit_time = entry_time + timedelta(hours=hold_hours)
            raw_pct = rng.gauss(0.0065, 0.024)
            if idx % 9 == 0:
                raw_pct -= rng.uniform(0.01, 0.035)
            pnl_pct = round(raw_pct, 4)
            direction_mult = 1 if direction == "LONG" else -1
            exit_price = round(entry_price * (1 + pnl_pct * direction_mult), 2)
            pnl = round((exit_price - entry_price) * qty * direction_mult - commission - slippage, 2)
            exit_reason = rng.choices(
                ["TAKE_PROFIT", "STOP_LOSS", "TIME_BARRIER", "MANUAL"],
                weights=[35, 28, 22, 15],
                k=1,
            )[0]
        trades.append(
            TradeRecord(
                id=str(uuid4()),
                symbol=symbol,
                direction=direction,
                status=status,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=commission,
                slippage=slippage,
                strategy=rng.choice(strategies),
                signal_confidence=confidence if status != "CANCELLED" else None,
                exit_reason=exit_reason,
                notes="Synthetic development trade",
            )
        )
    return tuple(sorted(trades, key=lambda trade: trade.entry_time, reverse=True))


def _normalize_live_trades(raw_trades: object) -> list[TradeRecord]:
    normalized: list[TradeRecord] = []
    if isinstance(raw_trades, dict):
        raw_iterable = list(raw_trades.values())
    else:
        raw_iterable = list(raw_trades) if raw_trades is not None else []
    for idx, item in enumerate(raw_iterable):
        order = getattr(item, "__dict__", item)
        status = str(order.get("status", "OPEN")).upper()
        fill_price = order.get("fill_price") or order.get("price")
        qty = float(order.get("quantity", 0.0))
        normalized.append(
            TradeRecord(
                id=str(order.get("id", f"live-{idx}")),
                symbol=str(order.get("symbol", "UNKNOWN")),
                direction="LONG" if qty >= 0 else "SHORT",
                status="CLOSED" if status == "FILLED" else ("CANCELLED" if status == "CANCELED" else "OPEN"),
                entry_time=datetime.now(timezone.utc),
                exit_time=None,
                entry_price=float(fill_price or 0.0),
                exit_price=None,
                quantity=abs(qty),
                pnl=None,
                pnl_pct=None,
                commission=0.0,
                slippage=0.0,
                strategy="LiveOrderManager",
                signal_confidence=None,
                exit_reason=None,
                notes="Live order manager fallback",
            )
        )
    return normalized


def load_trades() -> list[TradeRecord]:
    if not is_live_mode():
        return list(_mock_trades())

    parquet_path = Path("data/trades.parquet")
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
        return [_row_to_trade(row) for _, row in frame.iterrows()]

    try:
        from execution.order_manager import OrderManager
        from execution.paper_broker import PaperBroker

        manager = OrderManager(broker=PaperBroker(initial_cash=100_000))
        get_all = getattr(manager, "get_all_trades", None)
        raw = get_all() if callable(get_all) else manager.orders
        trades = _normalize_live_trades(raw)
        if trades:
            return trades
    except Exception:
        pass
    return list(_mock_trades())


def _row_to_trade(row: pd.Series) -> TradeRecord:
    payload = row.to_dict()
    return TradeRecord(
        id=str(payload.get("id", uuid4())),
        symbol=str(payload.get("symbol", "SPY")),
        direction=str(payload.get("direction", "LONG")).upper(),  # type: ignore[arg-type]
        status=str(payload.get("status", "CLOSED")).upper(),  # type: ignore[arg-type]
        entry_time=pd.Timestamp(payload.get("entry_time")).to_pydatetime().astimezone(timezone.utc),
        exit_time=(
            pd.Timestamp(payload["exit_time"]).to_pydatetime().astimezone(timezone.utc)
            if payload.get("exit_time") is not None
            else None
        ),
        entry_price=float(payload.get("entry_price", 0.0)),
        exit_price=float(payload["exit_price"]) if payload.get("exit_price") is not None else None,
        quantity=float(payload.get("quantity", 0.0)),
        pnl=float(payload["pnl"]) if payload.get("pnl") is not None else None,
        pnl_pct=float(payload["pnl_pct"]) if payload.get("pnl_pct") is not None else None,
        commission=float(payload.get("commission", 0.0)),
        slippage=float(payload.get("slippage", 0.0)),
        strategy=str(payload.get("strategy", "UnknownStrategy")),
        signal_confidence=float(payload["signal_confidence"]) if payload.get("signal_confidence") is not None else None,
        exit_reason=payload.get("exit_reason"),
        notes=payload.get("notes"),
    )


def filter_trades(
    trades: list[TradeRecord],
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strategy: Optional[str] = None,
) -> list[TradeRecord]:
    filtered = trades
    if symbol:
        filtered = [trade for trade in filtered if trade.symbol.upper() == symbol.upper()]
    if direction:
        filtered = [trade for trade in filtered if trade.direction == direction.upper()]
    if status:
        normalized_status = "CANCELLED" if status.upper() == "CANCELED" else status.upper()
        filtered = [trade for trade in filtered if trade.status == normalized_status]
    if start_date:
        start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
        filtered = [trade for trade in filtered if trade.entry_time >= start_dt]
    if end_date:
        end_dt = datetime.combine(end_date, time.max, tzinfo=timezone.utc)
        filtered = [trade for trade in filtered if trade.entry_time <= end_dt]
    if strategy:
        filtered = [trade for trade in filtered if trade.strategy == strategy]
    return filtered


def paginate_trades(trades: list[TradeRecord], page: int = 1, page_size: int = 25) -> dict[str, object]:
    total = len(trades)
    pages = max(1, math.ceil(total / page_size)) if total else 1
    page = max(1, min(page, pages))
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "trades": [trade.to_dict() for trade in trades[start:end]],
        "total": total,
        "page": page,
        "pages": pages,
    }


def get_trade(trade_id: str) -> Optional[TradeRecord]:
    for trade in load_trades():
        if trade.id == trade_id:
            return trade
    return None


def get_trade_summary(trades: Optional[list[TradeRecord]] = None) -> dict[str, float]:
    trade_list = trades or load_trades()
    today = datetime.now(timezone.utc).date()
    closed = [trade for trade in trade_list if trade.status == "CLOSED" and trade.pnl is not None]
    today_pnl = sum(trade.pnl or 0.0 for trade in closed if trade.exit_time and trade.exit_time.date() == today)
    wins = sum(1 for trade in closed if (trade.pnl or 0.0) > 0)
    return {
        "open": float(sum(1 for trade in trade_list if trade.status == "OPEN")),
        "closed": float(len(closed)),
        "today_pnl": round(today_pnl, 2),
        "total_pnl": round(sum(trade.pnl or 0.0 for trade in closed), 2),
        "win_rate": round((wins / len(closed)) if closed else 0.0, 4),
    }


def get_trade_price_context(trade_id: str) -> Optional[dict[str, object]]:
    trade = get_trade(trade_id)
    if trade is None:
        return None

    end_time = trade.exit_time or (trade.entry_time + timedelta(hours=2))
    start_time = trade.entry_time - timedelta(hours=2)
    current = start_time
    bars: list[dict[str, object]] = []
    rng = random.Random(trade.id)
    baseline = trade.entry_price
    while current <= end_time + timedelta(hours=2):
        drift = ((current - trade.entry_time).total_seconds() / 3600.0) * 0.12
        noise = rng.uniform(-1.5, 1.5)
        open_price = baseline + drift + noise
        close_price = open_price + rng.uniform(-1.2, 1.2)
        high = max(open_price, close_price) + rng.uniform(0.1, 0.8)
        low = min(open_price, close_price) - rng.uniform(0.1, 0.8)
        bars.append(
            {
                "time": current,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": round(rng.uniform(1_000, 12_000), 2),
            }
        )
        current += timedelta(minutes=15)
    return {
        "bars": bars,
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
    }


def get_trade_metadata(trades: Optional[list[TradeRecord]] = None) -> dict[str, list[str]]:
    trade_list = trades or load_trades()
    return {
        "symbols": sorted({trade.symbol for trade in trade_list}),
        "strategies": sorted({trade.strategy for trade in trade_list}),
    }

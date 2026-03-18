from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.metrics import summary

try:
    from api.services.trade_service import TradeRecord, filter_trades, load_trades
except ImportError:
    from services.trade_service import TradeRecord, filter_trades, load_trades


def _closed_trades(trades: list[TradeRecord]) -> list[TradeRecord]:
    return [trade for trade in trades if trade.status == "CLOSED" and trade.pnl is not None and trade.pnl_pct is not None]


def _equity_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    closed = _closed_trades(trades)
    if not closed:
        return pd.DataFrame(columns=["equity", "drawdown"], index=pd.DatetimeIndex([], tz="UTC"))
    rows = sorted(((trade.exit_time or trade.entry_time, trade.pnl or 0.0) for trade in closed), key=lambda row: row[0])
    timestamps = pd.DatetimeIndex([row[0] for row in rows])
    pnl = pd.Series([row[1] for row in rows], index=timestamps, dtype=float)
    # Lightweight Charts requires strictly increasing timestamps. Multiple trades
    # can close at the same instant, so aggregate identical timestamps first.
    pnl = pnl.groupby(level=0).sum().sort_index()
    equity = 100_000.0 + pnl.cumsum()
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame({"equity": equity, "drawdown": drawdown}, index=equity.index)


def get_trading_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    asset_class: Optional[str] = None,
) -> dict[str, object]:
    trades = filter_trades(
        load_trades(),
        symbol=symbol,
        strategy=strategy,
        asset_class=asset_class,
        start_date=start_date.date() if start_date else None,
        end_date=end_date.date() if end_date else None,
    )
    closed = _closed_trades(trades)
    equity_df = _equity_dataframe(trades)
    metric_summary = summary(equity_df["equity"], pd.Series([trade.pnl or 0.0 for trade in closed], dtype=float))
    wins = [trade for trade in closed if (trade.pnl or 0.0) > 0]
    losses = [trade for trade in closed if (trade.pnl or 0.0) < 0]
    holding_hours = [
        ((trade.exit_time or trade.entry_time) - trade.entry_time).total_seconds() / 3600.0
        for trade in closed
        if trade.exit_time is not None
    ]
    best_trade = max(closed, key=lambda trade: trade.pnl or float("-inf"), default=None)
    worst_trade = min(closed, key=lambda trade: trade.pnl or float("inf"), default=None)
    return {
        "total_trades": len(trades),
        "open_trades": sum(1 for trade in trades if trade.status == "OPEN"),
        "closed_trades": len(closed),
        "win_rate": round(metric_summary["win_rate"], 4),
        "avg_win_pct": round(float(np.mean([trade.pnl_pct for trade in wins])) if wins else 0.0, 4),
        "avg_loss_pct": round(float(np.mean([trade.pnl_pct for trade in losses])) if losses else 0.0, 4),
        "max_win_pct": round(max((trade.pnl_pct or 0.0) for trade in wins), 4) if wins else 0.0,
        "max_loss_pct": round(min((trade.pnl_pct or 0.0) for trade in losses), 4) if losses else 0.0,
        "profit_factor": round(float(metric_summary["profit_factor"]), 4),
        "sharpe_ratio": round(float(metric_summary["sharpe_ratio"]), 4),
        "max_drawdown_pct": round(float(metric_summary["max_drawdown"]), 4),
        "cagr": round(float(metric_summary["cagr"]), 4),
        "calmar_ratio": round(float(metric_summary["calmar_ratio"]), 4),
        "total_pnl": round(sum(trade.pnl or 0.0 for trade in closed), 2),
        "total_commission": round(sum(trade.commission for trade in trades), 2),
        "avg_holding_hours": round(float(np.mean(holding_hours)) if holding_hours else 0.0, 2),
        "best_trade": best_trade.to_dict() if best_trade else None,
        "worst_trade": worst_trade.to_dict() if worst_trade else None,
    }


def get_equity_curve(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    asset_class: Optional[str] = None,
) -> dict[str, list[dict[str, object]]]:
    trades = filter_trades(
        load_trades(),
        asset_class=asset_class,
        start_date=start_date.date() if start_date else None,
        end_date=end_date.date() if end_date else None,
    )
    equity_df = _equity_dataframe(trades)
    equity_only_df = _equity_dataframe(filter_trades(trades, asset_class="equity"))
    crypto_only_df = _equity_dataframe(filter_trades(trades, asset_class="crypto"))
    points = [
        {"timestamp": timestamp.to_pydatetime(), "equity": round(row["equity"], 2), "drawdown": round(row["drawdown"], 4)}
        for timestamp, row in equity_df.iterrows()
    ]
    equity_points = [
        {"timestamp": timestamp.to_pydatetime(), "equity": round(row["equity"], 2), "drawdown": round(row["drawdown"], 4)}
        for timestamp, row in equity_only_df.iterrows()
    ]
    crypto_points = [
        {"timestamp": timestamp.to_pydatetime(), "equity": round(row["equity"], 2), "drawdown": round(row["drawdown"], 4)}
        for timestamp, row in crypto_only_df.iterrows()
    ]
    benchmark: list[dict[str, object]] = []
    if points:
        start_equity = points[0]["equity"]
        for idx, point in enumerate(points):
            benchmark_equity = start_equity * (1 + 0.0009 * idx + math.sin(idx / 5.0) * 0.002)
            benchmark.append(
                {
                    "timestamp": point["timestamp"],
                    "equity": round(benchmark_equity, 2),
                    "drawdown": round(min(0.0, math.sin(idx / 4.0) * 0.03), 4),
                }
            )
    return {
        "points": points,
        "benchmark": benchmark,
        "equity_points": equity_points,
        "crypto_points": crypto_points,
        "combined_points": points,
    }


def get_pnl_distribution(asset_class: Optional[str] = None) -> dict[str, list[dict[str, object]]]:
    closed = _closed_trades(filter_trades(load_trades(), asset_class=asset_class))
    values = [trade.pnl_pct or 0.0 for trade in closed]
    if not values:
        return {"buckets": []}
    bins = np.arange(-0.05, 0.055, 0.005)
    hist, edges = np.histogram(values, bins=bins)
    total = len(values)
    buckets = []
    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        buckets.append(
            {
                "range": f"{left * 100:.1f}% to {right * 100:.1f}%",
                "count": int(count),
                "pct": round(count / total, 4),
            }
        )
    return {"buckets": buckets}


def get_performance_by_symbol(asset_class: Optional[str] = None) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in filter_trades(load_trades(), asset_class=asset_class):
        grouped[trade.symbol].append(trade)
    rows = []
    for symbol, trades in grouped.items():
        closed = _closed_trades(trades)
        equity_df = _equity_dataframe(trades)
        returns = equity_df["equity"].pct_change().dropna()
        sharpe = 0.0 if returns.empty else float(np.sqrt(252.0) * returns.mean() / returns.std(ddof=0)) if returns.std(ddof=0) else 0.0
        rows.append(
            {
                "symbol": symbol,
                "asset_class": trades[0].asset_class if trades else "equity",
                "trades": len(closed),
                "win_rate": round(sum(1 for trade in closed if (trade.pnl or 0.0) > 0) / len(closed), 4) if closed else 0.0,
                "avg_pnl": round(float(np.mean([trade.pnl_pct for trade in closed])) if closed else 0.0, 4),
                "total_pnl": round(sum(trade.pnl or 0.0 for trade in closed), 2),
                "sharpe": round(sharpe, 4),
            }
        )
    rows.sort(key=lambda row: row["total_pnl"], reverse=True)
    return {"rows": rows}


def get_performance_by_exit_reason(asset_class: Optional[str] = None) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in _closed_trades(filter_trades(load_trades(), asset_class=asset_class)):
        grouped[trade.exit_reason or "UNKNOWN"].append(trade)
    rows = []
    for exit_reason, trades in grouped.items():
        rows.append(
            {
                "exit_reason": exit_reason,
                "trades": len(trades),
                "win_rate": round(sum(1 for trade in trades if (trade.pnl or 0.0) > 0) / len(trades), 4) if trades else 0.0,
                "avg_pnl": round(float(np.mean([trade.pnl_pct for trade in trades])) if trades else 0.0, 4),
                "total_pnl": round(sum(trade.pnl or 0.0 for trade in trades), 2),
            }
        )
    rows.sort(key=lambda row: row["trades"], reverse=True)
    return {"rows": rows}

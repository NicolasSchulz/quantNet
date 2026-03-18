from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtesting.cost_model import CostModel
from backtesting.metrics import summary
from risk.position_sizer import VolatilityParitySizer
from strategies.base_strategy import BaseStrategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    regime_status: pd.Series | None = None
    strategy_metadata: dict = field(default_factory=dict)


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    naive = index.tz_localize(None) if index.tz is not None else index
    return index.to_series().groupby(naive.to_period(freq)).tail(1).index


def _update_position(prev_qty: float, prev_avg: float, trade_qty: float, fill_price: float) -> tuple[float, float, float]:
    """Update qty/avg cost and return realized PnL from this trade."""
    if trade_qty == 0:
        return prev_qty, prev_avg, 0.0

    realized = 0.0
    new_qty = prev_qty + trade_qty

    # Increasing existing side or opening new position
    if prev_qty == 0 or (prev_qty > 0 and trade_qty > 0) or (prev_qty < 0 and trade_qty < 0):
        total_abs = abs(prev_qty) + abs(trade_qty)
        new_avg = (
            0.0
            if total_abs == 0
            else ((abs(prev_qty) * prev_avg) + (abs(trade_qty) * fill_price)) / total_abs
        )
        return new_qty, new_avg, realized

    # Reducing or flipping
    closing_qty = min(abs(prev_qty), abs(trade_qty))
    if prev_qty > 0 and trade_qty < 0:
        realized += closing_qty * (fill_price - prev_avg)
    elif prev_qty < 0 and trade_qty > 0:
        realized += closing_qty * (prev_avg - fill_price)

    if new_qty == 0:
        return 0.0, 0.0, realized

    # Flip case: leftover opens opposite side at current fill
    if (prev_qty > 0 > new_qty) or (prev_qty < 0 < new_qty):
        return new_qty, fill_price, realized

    # Still on same side as before, avg cost unchanged
    return new_qty, prev_avg, realized


def run(
    strategy: BaseStrategy,
    data: dict[str, pd.DataFrame],
    initial_capital: float,
    cost_model: CostModel,
    rebalance_frequency: str = "M",
    sizing_method: str = "equal_weight",
    volatility_sizer: VolatilityParitySizer | None = None,
    trading_hours: str | None = None,
    rebalance_freq: str | None = None,
) -> BacktestResult:
    if not data:
        raise ValueError("Backtest data cannot be empty")

    inferred_asset_class = str(next(iter(data.values())).get("asset_class", pd.Series(["equity"])).iloc[0]).lower()
    effective_trading_hours = trading_hours or ("24/7" if inferred_asset_class == "crypto" else "market")
    _ = effective_trading_hours
    effective_rebalance = rebalance_freq or rebalance_frequency

    open_prices = pd.concat(
        {sym: df["open"].astype(float) for sym, df in data.items()}, axis=1
    ).dropna(how="any")
    close_prices = pd.concat(
        {sym: df["close"].astype(float) for sym, df in data.items()}, axis=1
    ).reindex(open_prices.index).dropna(how="any")

    if len(open_prices) < 3:
        raise ValueError("Need at least 3 aligned bars for backtest.")

    if bool(getattr(strategy, "expects_ohlcv", False)):
        if len(data) != 1:
            raise ValueError("Strategies requiring OHLCV currently support single-symbol backtests only.")
        only_symbol = list(data.keys())[0]
        raw_input = data[only_symbol][["open", "high", "low", "close", "volume"]].reindex(close_prices.index)
    else:
        raw_input = close_prices

    raw_signals = strategy.generate_signals(raw_input)
    if raw_signals.empty:
        raise ValueError("Strategy returned no signals.")

    if isinstance(raw_signals.index, pd.MultiIndex):
        signal_matrix = raw_signals.unstack("symbol").reindex(close_prices.index).fillna(0)
    else:
        if isinstance(raw_signals, pd.Series):
            if raw_signals.name in close_prices.columns:
                signal_matrix = raw_signals.to_frame(name=str(raw_signals.name))
            elif len(close_prices.columns) == 1:
                signal_matrix = raw_signals.to_frame(name=close_prices.columns[0])
            else:
                signal_matrix = pd.DataFrame(raw_signals)
        else:
            signal_matrix = pd.DataFrame(raw_signals)
        signal_matrix = signal_matrix.reindex(close_prices.index).fillna(0)

    signal_matrix = signal_matrix.reindex(columns=close_prices.columns, fill_value=0).astype(float)
    returns = close_prices.pct_change().fillna(0.0)
    warmup_bars = int(strategy.warmup_bars()) if hasattr(strategy, "warmup_bars") else 0

    rebal_dates = set(_rebalance_dates(close_prices.index, effective_rebalance))
    sizer = volatility_sizer or VolatilityParitySizer()

    shares = {sym: 0.0 for sym in close_prices.columns}
    avg_cost = {sym: 0.0 for sym in close_prices.columns}
    cash = float(initial_capital)
    equity = []
    trades: list[dict[str, float | str | pd.Timestamp]] = []

    # Evaluate equity for first bar without trades.
    first_value = cash + sum(shares[s] * close_prices.iloc[0][s] for s in shares)
    equity.append((close_prices.index[0], first_value))

    for i in range(0, len(close_prices.index) - 1):
        current_dt = close_prices.index[i]
        next_dt = close_prices.index[i + 1]

        if i >= warmup_bars and current_dt in rebal_dates:
            row_signals = signal_matrix.loc[current_dt]
            if sizing_method == "equal_weight":
                abs_sum = row_signals.abs().sum()
                if abs_sum > 0:
                    target_weights = row_signals / abs_sum
                else:
                    target_weights = row_signals * 0.0
            elif sizing_method == "volatility_parity":
                target_weights = sizer.compute_weights(
                    signals=row_signals,
                    returns=returns,
                    date=current_dt,
                ).reindex(row_signals.index).fillna(0.0)
            else:
                raise ValueError(
                    "sizing_method must be 'equal_weight' or 'volatility_parity'"
                )

            current_equity = cash + sum(shares[s] * close_prices.loc[current_dt, s] for s in shares)

            for sym in close_prices.columns:
                next_open = float(open_prices.loc[next_dt, sym])
                target_value = float(target_weights.get(sym, 0.0) * current_equity)
                target_qty = target_value / next_open if next_open > 0 else 0.0
                trade_qty = target_qty - shares[sym]

                if abs(trade_qty) < 1e-10:
                    continue

                direction = 1 if trade_qty > 0 else -1
                fill_price = cost_model.apply(next_open, abs(trade_qty), direction)
                notional = abs(trade_qty) * fill_price
                commission = cost_model.calculate_commission(notional)

                cash -= trade_qty * fill_price
                cash -= commission

                new_qty, new_avg, realized = _update_position(
                    shares[sym], avg_cost[sym], trade_qty, fill_price
                )
                shares[sym] = new_qty
                avg_cost[sym] = new_avg

                trades.append(
                    {
                        "timestamp": next_dt,
                        "symbol": sym,
                        "quantity": trade_qty,
                        "fill_price": fill_price,
                        "commission": commission,
                        "realized_pnl": realized,
                    }
                )

        mark_dt = close_prices.index[i + 1]
        portfolio_value = cash + sum(shares[s] * close_prices.loc[mark_dt, s] for s in shares)
        equity.append((mark_dt, portfolio_value))

    equity_curve = pd.Series(
        [v for _, v in equity], index=pd.DatetimeIndex([d for d, _ in equity], tz="UTC"), name="equity"
    )
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=["timestamp", "symbol", "quantity", "fill_price", "commission", "realized_pnl"]
        )

    stats = summary(equity_curve=equity_curve, trades=trades_df["realized_pnl"])
    regime_status = getattr(strategy, "regime_series", None)
    strategy_metadata = strategy.get_parameters() if hasattr(strategy, "get_parameters") else {}
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades_df,
        metrics=stats,
        regime_status=regime_status,
        strategy_metadata=strategy_metadata,
    )

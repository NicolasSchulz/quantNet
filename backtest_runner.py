from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from backtesting.cost_model import CostModel
from backtesting.engine import run
from backtesting.reporter import plot_equity_curve, print_metrics
from data.ingestion.yahoo_feed import YahooFeed
from data.normalizer import normalize_ohlcv
from strategies.examples.simple_momentum import SimpleMomentumStrategy


def load_settings(path: str = "config/settings.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_universe(feed: YahooFeed, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, interval: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = feed.fetch_historical(symbol=symbol, start=start, end=end, interval=interval)
        normalized = normalize_ohlcv(raw, symbol=symbol, asset_class="etf")
        data[symbol] = normalized
    return data


def main() -> None:
    settings = load_settings()
    initial_capital = float(settings["backtesting"]["initial_capital"])
    cost_cfg = settings["backtesting"]["cost_model"]
    interval = settings["data"]["default_interval"]

    cost_model = CostModel(
        commission=float(cost_cfg["commission"]),
        slippage_bps=float(cost_cfg["slippage_bps"]),
        spread_bps=float(cost_cfg["spread_bps"]),
    )

    symbols = ["SPY", "QQQ", "EEM", "GLD", "TLT"]
    end = pd.Timestamp.utcnow()
    start = end - pd.DateOffset(years=5)

    feed = YahooFeed()
    universe_data = fetch_universe(feed, symbols, start=start, end=end, interval=interval)

    strategy = SimpleMomentumStrategy(rebalance_frequency="M")
    result = run(
        strategy=strategy,
        data=universe_data,
        initial_capital=initial_capital,
        cost_model=cost_model,
        rebalance_frequency="M",
    )

    print_metrics(result.metrics, len(result.trades))
    out = plot_equity_curve(result.equity_curve, output_path="equity_curve.png")
    print(f"Equity curve saved to: {out}")


if __name__ == "__main__":
    main()

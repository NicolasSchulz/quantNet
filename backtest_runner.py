from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from backtesting.cost_model import CostModel
from backtesting.engine import BacktestResult, run
from backtesting.optimizer import LookbackOptimizer
from data.ingestion.yahoo_feed import YahooFeed
from data.normalizer import normalize_ohlcv
from data.storage.parquet_store import ParquetStore
from data.universes.etf_universe import EtfUniverse
from strategies.examples.simple_momentum import SimpleMomentumStrategy


def load_settings(path: str = "config/settings.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_or_fetch_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    feed: YahooFeed,
    store: ParquetStore,
) -> pd.DataFrame:
    symbol = symbol.upper()
    if store.exists(symbol, interval):
        cached = store.load(symbol=symbol, interval=interval, start=start, end=end)
        if not cached.empty:
            return cached

    raw = feed.fetch_historical(symbol=symbol, start=start, end=end, interval=interval)
    normalized = normalize_ohlcv(raw, symbol=symbol, asset_class="etf")
    store.save(normalized, symbol=symbol, interval=interval)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    return normalized[(normalized.index >= start_ts) & (normalized.index <= end_ts)]


def fetch_universe_cached(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    feed: YahooFeed,
    store: ParquetStore,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        data[symbol] = _load_or_fetch_symbol(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            feed=feed,
            store=store,
        )
    return data


def _print_table(rows: list[dict[str, str]]) -> None:
    headers = ["Szenario", "Sharpe", "CAGR", "Max DD", "Calmar", "Trades"]
    widths = {h: max(len(h), *(len(row[h]) for row in rows)) for h in headers}

    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


def _format_result_row(label: str, result: BacktestResult) -> dict[str, str]:
    return {
        "Szenario": label,
        "Sharpe": f"{result.metrics.get('sharpe_ratio', 0.0):.2f}",
        "CAGR": f"{result.metrics.get('cagr', 0.0):.2%}",
        "Max DD": f"{result.metrics.get('max_drawdown', 0.0):.2%}",
        "Calmar": f"{result.metrics.get('calmar_ratio', 0.0):.2f}",
        "Trades": str(len(result.trades)),
    }


def _plot_comparison(results: dict[str, BacktestResult], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, result in results.items():
        result.equity_curve.rename(label).plot(ax=ax, lw=1.8)
    ax.set_title("Equity Curves Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run momentum backtests with optional optimization")
    parser.add_argument("--optimize", action="store_true", help="Print full lookback optimization table")
    args = parser.parse_args()

    settings = load_settings()

    initial_capital = float(settings["backtesting"]["initial_capital"])
    interval = settings["data"]["default_interval"]
    lookback_years = int(settings["data"].get("lookback_years", 5))
    cost_cfg = settings["backtesting"]["cost_model"]
    strategy_cfg = settings["strategy"]
    optimizer_cfg = settings["optimizer"]
    vol_cfg = settings["risk"]["vol_parity"]

    cost_model = CostModel(
        commission=float(cost_cfg["commission"]),
        slippage_bps=float(cost_cfg["slippage_bps"]),
        spread_bps=float(cost_cfg["spread_bps"]),
    )

    end = pd.Timestamp.utcnow()
    start = end - pd.DateOffset(years=lookback_years)

    feed = YahooFeed()
    store = ParquetStore(storage_path=settings["data"]["storage_path"])
    universe = EtfUniverse()

    original_symbols = ["SPY", "QQQ", "EEM", "GLD", "TLT"]
    broad_symbols = universe.get_symbols()

    data_a = fetch_universe_cached(original_symbols, start, end, interval, feed, store)
    data_b = fetch_universe_cached(broad_symbols, start, end, interval, feed, store)

    # Regime filter benchmark data for scenarios C and D (helper symbol, not traded).
    data_cd = dict(data_b)
    regime_benchmark = str(strategy_cfg.get("regime_benchmark", "SPY")).upper()
    if regime_benchmark not in data_cd:
        data_cd[regime_benchmark] = _load_or_fetch_symbol(
            symbol=regime_benchmark,
            start=start,
            end=end,
            interval=interval,
            feed=feed,
            store=store,
        )

    optimizer = LookbackOptimizer(
        strategy_class=SimpleMomentumStrategy,
        data=data_cd,
        cost_model=cost_model,
        initial_capital=initial_capital,
        train_fraction=float(optimizer_cfg.get("train_fraction", 0.70)),
        overfit_ratio_warning=float(optimizer_cfg.get("overfit_ratio_warning", 1.5)),
        sizing_method="equal_weight",
    )

    param_grid = optimizer_cfg["param_grid"]
    optimization_df = optimizer.optimize_grid(param_grid)
    optimization_df.to_csv("optimization_results.csv", index=False)

    if args.optimize:
        print("\nOptimization Results (sorted by out-of-sample Sharpe):")
        print(optimization_df.to_string(index=False))

    best_row = optimization_df.iloc[0]
    best_params = dict(best_row["params"])

    train_sharpe = float(best_row.get("train_sharpe", 0.0))
    test_sharpe = float(best_row.get("sharpe", 0.0))
    if test_sharpe <= 0 and train_sharpe > 0:
        print("WARNING: Possible overfitting (positive train Sharpe with non-positive test Sharpe).")
    elif test_sharpe > 0 and train_sharpe / test_sharpe > float(optimizer_cfg.get("overfit_ratio_warning", 1.5)):
        print(
            "WARNING: Possible overfitting detected "
            f"(train Sharpe {train_sharpe:.2f} >> test Sharpe {test_sharpe:.2f})."
        )

    scenario_a = run(
        strategy=SimpleMomentumStrategy(
            formation_months=12,
            skip_months=1,
            rebalance_freq="M",
            use_regime_filter=False,
            tradable_symbols=original_symbols,
        ),
        data=data_a,
        initial_capital=initial_capital,
        cost_model=cost_model,
        rebalance_frequency="M",
        sizing_method="equal_weight",
    )

    scenario_b = run(
        strategy=SimpleMomentumStrategy(
            formation_months=12,
            skip_months=1,
            rebalance_freq="M",
            use_regime_filter=False,
            tradable_symbols=broad_symbols,
        ),
        data=data_b,
        initial_capital=initial_capital,
        cost_model=cost_model,
        rebalance_frequency="M",
        sizing_method="equal_weight",
    )

    scenario_c = run(
        strategy=SimpleMomentumStrategy(
            formation_months=12,
            skip_months=1,
            rebalance_freq="M",
            use_regime_filter=True,
            regime_benchmark=regime_benchmark,
            regime_ma_window=int(strategy_cfg.get("regime_ma_window", 200)),
            tradable_symbols=broad_symbols,
        ),
        data=data_cd,
        initial_capital=initial_capital,
        cost_model=cost_model,
        rebalance_frequency="M",
        sizing_method="equal_weight",
    )

    scenario_d = run(
        strategy=SimpleMomentumStrategy(
            formation_months=int(best_params.get("formation_months", 12)),
            skip_months=int(best_params.get("skip_months", 1)),
            rebalance_freq=str(best_params.get("rebalance_freq", "M")),
            use_regime_filter=True,
            regime_benchmark=regime_benchmark,
            regime_ma_window=int(strategy_cfg.get("regime_ma_window", 200)),
            tradable_symbols=broad_symbols,
        ),
        data=data_cd,
        initial_capital=initial_capital,
        cost_model=cost_model,
        rebalance_frequency=str(best_params.get("rebalance_freq", "M")),
        sizing_method="volatility_parity",
    )

    results = {
        "A": scenario_a,
        "B": scenario_b,
        "C": scenario_c,
        "D": scenario_d,
    }

    rows = [_format_result_row(label, result) for label, result in results.items()]
    print("\nSzenario-Vergleich:")
    _print_table(rows)

    _plot_comparison(results, output_path="equity_curves_comparison.png")
    print("\nSaved: equity_curves_comparison.png")
    print("Saved: optimization_results.csv")


if __name__ == "__main__":
    main()

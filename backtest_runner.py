from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from backtesting.cost_model import CostModel
from backtesting.engine import BacktestResult, run
from backtesting.optimizer import LookbackOptimizer
from data.ingestion.feed_factory import FeedFactory
from data.normalizer import normalize_ohlcv
from data.storage.parquet_store import ParquetStore
from data.universes.etf_universe import EtfUniverse
from strategies.examples.simple_momentum import SimpleMomentumStrategy
from strategies.filters.regime_filter import RegimeFilter

MIN_SYMBOLS_PER_SCENARIO = 3


def load_settings(path: str = "config/settings.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_or_fetch_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    feed,
    store: ParquetStore,
) -> pd.DataFrame:
    symbol = symbol.upper()
    if store.exists(symbol, interval):
        cached = store.load(symbol=symbol, interval=interval, start=start, end=end)
        if not cached.empty:
            return cached

    raw = feed.fetch_historical(symbol=symbol, start=start, end=end, interval=interval)
    normalized = normalize_ohlcv(raw, symbol=symbol, asset_class="etf", interval=interval)
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
    feed,
    store: ParquetStore,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for symbol in symbols:
        try:
            symbol_df = _load_or_fetch_symbol(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
                feed=feed,
                store=store,
            )
        except (LookupError, RuntimeError, FileNotFoundError, ValueError) as exc:
            failures.append(f"{symbol}: {exc}")
            continue

        if symbol_df.empty:
            failures.append(f"{symbol}: empty dataset after cache/fetch filtering")
            continue
        data[symbol] = symbol_df

    if failures:
        print("\nWARNING: Some symbols could not be loaded:")
        for item in failures:
            print(f"- {item}")
    return data


def _require_minimum_data(data: dict[str, pd.DataFrame], scenario_name: str, min_symbols: int) -> None:
    if len(data) < min_symbols:
        loaded = ", ".join(sorted(data.keys())) if data else "none"
        raise RuntimeError(
            f"{scenario_name}: too few symbols with usable data ({len(data)} < {min_symbols}). "
            f"Loaded: {loaded}. "
            "Check network/Yahoo availability or use an existing Parquet cache."
        )


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
    interval = settings["data"]["intervals"]["primary"]
    daily_interval = settings["data"]["intervals"]["daily"]
    lookback_years = int(settings["data"].get("lookback_years", 5))
    cost_cfg = settings["backtesting"]["cost_model"]
    strategy_cfg = settings["strategy"]
    optimizer_cfg = settings["optimizer"]
    cost_model = CostModel(
        commission=float(cost_cfg["commission"]),
        slippage_bps=float(cost_cfg["slippage_bps"]),
        spread_bps=float(cost_cfg["spread_bps"]),
    )

    end = pd.Timestamp.utcnow()
    start = end - pd.DateOffset(years=lookback_years)

    feed, store = FeedFactory.create_with_cache(config=settings)
    universe = EtfUniverse()

    original_symbols = ["SPY", "QQQ", "EEM", "GLD", "TLT"]
    broad_symbols = universe.get_symbols()

    data_a = fetch_universe_cached(original_symbols, start, end, interval, feed, store)
    data_b = fetch_universe_cached(broad_symbols, start, end, interval, feed, store)
    _require_minimum_data(data_a, "Szenario A", MIN_SYMBOLS_PER_SCENARIO)
    _require_minimum_data(data_b, "Szenario B/C/D", MIN_SYMBOLS_PER_SCENARIO)

    # Regime filter benchmark data for scenarios C and D (helper symbol, not traded).
    data_cd = dict(data_b)
    regime_benchmark = str(strategy_cfg.get("regime_benchmark", "SPY")).upper()
    if regime_benchmark not in data_cd:
        try:
            data_cd[regime_benchmark] = _load_or_fetch_symbol(
                symbol=regime_benchmark,
                start=start,
                end=end,
                interval=interval,
                feed=feed,
                store=store,
            )
        except (LookupError, RuntimeError, FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"Regime benchmark '{regime_benchmark}' could not be loaded: {exc}. "
                "Scenario C/D requires this benchmark for the regime filter."
            ) from exc

    # 1h strategy bars + daily regime benchmark (200d MA) projected onto 1h index.
    daily_regime_series = None
    try:
        spy_daily = _load_or_fetch_symbol(
            symbol=regime_benchmark,
            start=start,
            end=end,
            interval=daily_interval,
            feed=feed,
            store=store,
        )
        regime_filter_daily = RegimeFilter(benchmark=regime_benchmark, ma_window=int(strategy_cfg.get("regime_ma_window", 200)))
        daily_regime = regime_filter_daily.get_regime_series(
            spy_daily[["close"]].rename(columns={"close": regime_benchmark})
        )
        ref_index = list(data_cd.values())[0].index
        daily_regime_series = daily_regime.reindex(ref_index).ffill().fillna(False)
    except Exception as exc:
        print(f"WARNING: Daily regime benchmark unavailable, fallback to in-timeframe regime ({exc})")

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
            external_regime_series=daily_regime_series,
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
            external_regime_series=daily_regime_series,
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

    # Scenario E: ML strategy on SPY via registry default model.
    ml_cfg = settings.get("ml", {})
    registry_path = ml_cfg.get("registry", {}).get("path", "./models/registry.json")
    ml_strategy_cfg = ml_cfg.get("strategy", {})
    if "SPY" in data_cd:
        try:
            from strategies.filters.regime_filter import RegimeFilter
            from strategies.ml.feature_engineer import FeatureEngineer
            from strategies.ml.ml_strategy import MLStrategy
            from strategies.ml.model_registry import ModelNotFoundError, ModelRegistry
            from strategies.ml.signal_filter import SignalFilter

            registry = ModelRegistry(registry_path=str(registry_path))
            signal_filter = SignalFilter(
                min_confidence=float(ml_strategy_cfg.get("min_confidence", 0.45)),
                regime_filter=RegimeFilter() if bool(ml_strategy_cfg.get("use_regime_filter", True)) else None,
                min_holding_days=int(ml_strategy_cfg.get("min_holding_days", 3)),
                signal_smoothing=bool(ml_strategy_cfg.get("signal_smoothing", True)),
            )
            ml_strategy = MLStrategy(
                symbol="SPY",
                model_registry=registry,
                feature_engineer=FeatureEngineer(),
                signal_filter=signal_filter,
            )
            scenario_e = run(
                strategy=ml_strategy,
                data={"SPY": data_cd["SPY"]},
                initial_capital=initial_capital,
                cost_model=cost_model,
                rebalance_frequency="D",
                sizing_method="equal_weight",
            )
            results["E (ML)"] = scenario_e
        except Exception as exc:
            if "No default model set" in str(exc) or "not found in registry" in str(exc):
                print("\nWARNING: Kein trainiertes Modell gefunden. Fuehre model_training.py aus.")
            else:
                print(f"\nWARNING: Scenario E (ML) skipped due to error: {exc}")
    else:
        print("\nWARNING: Scenario E (ML) skipped because SPY data is unavailable.")

    rows = [_format_result_row(label, result) for label, result in results.items()]
    print("\nSzenario-Vergleich:")
    _print_table(rows)

    _plot_comparison(results, output_path="equity_curves_comparison.png")
    print("\nSaved: equity_curves_comparison.png")
    print("Saved: optimization_results.csv")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1) from exc

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
from data.universes.crypto_universe import CryptoUniverse
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
    asset_class: str = "equity",
) -> pd.DataFrame:
    symbol = symbol.upper()
    start_str = pd.Timestamp(start).date().isoformat()
    end_str = pd.Timestamp(end).date().isoformat()
    if store.exists(symbol, interval):
        cached = store.load(symbol=symbol, interval=interval, start=start, end=end)
        requested_end = pd.Timestamp(end)
        if requested_end.tzinfo is None:
            requested_end = requested_end.tz_localize("UTC")
        else:
            requested_end = requested_end.tz_convert("UTC")
        if not cached.empty and cached.index.max() >= requested_end - pd.Timedelta(days=1):
            return cached

    raw = feed.fetch_historical(symbol=symbol, start=start_str, end=end_str, interval=interval)
    normalized = normalize_ohlcv(raw, symbol=symbol, asset_class=asset_class, interval=interval)
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
    asset_class: str = "equity",
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for symbol in symbols:
        try:
            try:
                symbol_df = _load_or_fetch_symbol(
                    symbol=symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    feed=feed,
                    store=store,
                    asset_class=asset_class,
                )
            except TypeError as exc:
                if "asset_class" not in str(exc):
                    raise
                symbol_df = _load_or_fetch_symbol(symbol, start, end, interval, feed, store)
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
    if not rows:
        print("Keine Szenarien mit Ergebnissen.")
        return
    headers = ["Szenario", "Asset Class", "Sharpe", "CAGR", "Max DD", "Calmar", "Trades"]
    widths = {h: max(len(h), *(len(row[h]) for row in rows)) for h in headers}

    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


def _format_result_row(label: str, result: BacktestResult, asset_class: str) -> dict[str, str]:
    return {
        "Szenario": label,
        "Asset Class": asset_class.title(),
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
    parser.add_argument(
        "--scenario",
        choices=["ALL", "A", "B", "C", "D", "E", "F", "G"],
        default="ALL",
        help="Run only a single scenario. G = BTC ML holdout backtest.",
    )
    args = parser.parse_args()

    settings = load_settings()

    initial_capital = float(settings["backtesting"]["initial_capital"])
    interval = settings["data"]["intervals"]["primary"]
    daily_interval = settings["data"]["intervals"]["daily"]
    lookback_years = int(settings["data"].get("lookback_years", 5))
    cost_cfg = settings["backtesting"]["cost_model"]
    strategy_cfg = settings["strategy"]
    optimizer_cfg = settings["optimizer"]
    equity_cost_model = CostModel(
        asset_class="equity",
        commission=float(cost_cfg["equity"]["commission"]),
        slippage_bps=float(cost_cfg["equity"]["slippage_bps"]),
        spread_bps=float(cost_cfg["equity"]["spread_bps"]),
    )
    crypto_cost_model = CostModel(
        asset_class="crypto",
        commission=float(cost_cfg["crypto"]["commission"]),
        slippage_bps=float(cost_cfg["crypto"]["slippage_bps"]),
        spread_bps=float(cost_cfg["crypto"]["spread_bps"]),
    )

    end = pd.Timestamp.utcnow()
    start = end - pd.DateOffset(years=lookback_years)

    feed, store = FeedFactory.create_with_cache(config=settings)
    universe = EtfUniverse()
    crypto_universe = CryptoUniverse()

    original_symbols = ["SPY", "QQQ", "EEM", "GLD", "TLT"]
    broad_symbols = universe.get_symbols()

    run_equity_block = args.scenario in {"ALL", "A", "B", "C", "D", "E"}
    run_crypto_block = args.scenario in {"ALL", "F", "G"}

    data_a: dict[str, pd.DataFrame] = {}
    data_b: dict[str, pd.DataFrame] = {}
    if run_equity_block:
        data_a = fetch_universe_cached(original_symbols, start, end, interval, feed, store, asset_class="equity")
        data_b = fetch_universe_cached(broad_symbols, start, end, interval, feed, store, asset_class="equity")
        if args.scenario in {"ALL", "A"}:
            _require_minimum_data(data_a, "Szenario A", MIN_SYMBOLS_PER_SCENARIO)
        if args.scenario in {"ALL", "B", "C", "D", "E"}:
            _require_minimum_data(data_b, "Szenario B/C/D", MIN_SYMBOLS_PER_SCENARIO)

    # Regime filter benchmark data for scenarios C and D (helper symbol, not traded).
    data_cd = dict(data_b)
    regime_benchmark = str(strategy_cfg.get("regime_benchmark", "SPY")).upper()
    if run_equity_block and regime_benchmark not in data_cd:
        try:
            data_cd[regime_benchmark] = _load_or_fetch_symbol(
                symbol=regime_benchmark,
                asset_class="equity",
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
    if run_equity_block:
        try:
            spy_daily = _load_or_fetch_symbol(
                symbol=regime_benchmark,
                asset_class="equity",
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
    best_params = {"formation_months": 12, "skip_months": 1, "rebalance_freq": "M"}
    if run_equity_block:
        optimizer = LookbackOptimizer(
            strategy_class=SimpleMomentumStrategy,
            data=data_cd,
            cost_model=equity_cost_model,
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

    results: dict[str, tuple[str, BacktestResult]] = {}
    if args.scenario in {"ALL", "A"}:
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
            cost_model=equity_cost_model,
            rebalance_frequency="M",
            sizing_method="equal_weight",
        )
        results["A"] = ("equity", scenario_a)

    if args.scenario in {"ALL", "B"}:
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
            cost_model=equity_cost_model,
            rebalance_frequency="M",
            sizing_method="equal_weight",
        )
        results["B"] = ("equity", scenario_b)

    if args.scenario in {"ALL", "C"}:
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
            cost_model=equity_cost_model,
            rebalance_frequency="M",
            sizing_method="equal_weight",
        )
        results["C"] = ("equity", scenario_c)

    if args.scenario in {"ALL", "D"}:
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
            cost_model=equity_cost_model,
            rebalance_frequency=str(best_params.get("rebalance_freq", "M")),
            sizing_method="volatility_parity",
        )
        results["D"] = ("equity", scenario_d)

    # Scenario E: ML strategy on SPY via registry default model.
    ml_cfg = settings.get("ml", {})
    dataset_cfg = ml_cfg.get("dataset", {})
    registry_path = ml_cfg.get("registry", {}).get("path", "./models/registry.json")
    ml_strategy_cfg = ml_cfg.get("strategy", {})
    train_end = pd.Timestamp(dataset_cfg.get("train_end", "2025-12-31"), tz="UTC")
    holdout_start = pd.Timestamp(dataset_cfg.get("holdout_start", "2026-01-01"), tz="UTC")
    if args.scenario in {"ALL", "E"} and "SPY" in data_cd:
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
                feature_engineer=FeatureEngineer(asset_class="equity"),
                signal_filter=signal_filter,
            )
            scenario_e = run(
                strategy=ml_strategy,
                data={"SPY": data_cd["SPY"]},
                initial_capital=initial_capital,
                cost_model=equity_cost_model,
                rebalance_frequency="D",
                sizing_method="equal_weight",
            )
            results["E (ML)"] = ("equity", scenario_e)
        except Exception as exc:
            if "No default model set" in str(exc) or "not found in registry" in str(exc):
                print("\nWARNING: Kein trainiertes Modell gefunden. Fuehre model_training.py aus.")
            else:
                print(f"\nWARNING: Scenario E (ML) skipped due to error: {exc}")
    elif args.scenario in {"ALL", "E"}:
        print("\nWARNING: Scenario E (ML) skipped because SPY data is unavailable.")

    crypto_symbols = crypto_universe.get_symbols()
    crypto_end = pd.Timestamp.utcnow()
    crypto_start = max(crypto_end - pd.DateOffset(years=3), pd.Timestamp("2020-09-01", tz="UTC"))
    data_f: dict[str, pd.DataFrame] = {}
    if run_crypto_block:
        crypto_feed, crypto_store = FeedFactory.create_with_cache(config=settings, source="binance", symbol="BTCUSDT")
        data_f = fetch_universe_cached(crypto_symbols, crypto_start, crypto_end, interval, crypto_feed, crypto_store, asset_class="crypto")

    if args.scenario in {"ALL", "F"}:
        try:
            _require_minimum_data(data_f, "Szenario F", MIN_SYMBOLS_PER_SCENARIO)
            scenario_f = run(
                strategy=SimpleMomentumStrategy(
                    formation_months=12,
                    skip_months=1,
                    rebalance_freq="D",
                    use_regime_filter=False,
                    tradable_symbols=crypto_symbols,
                ),
                data=data_f,
                initial_capital=initial_capital,
                cost_model=crypto_cost_model,
                rebalance_frequency="D",
                sizing_method="equal_weight",
                trading_hours="24/7",
            )
            results["F"] = ("crypto", scenario_f)
        except Exception as exc:
            print(f"\nWARNING: Scenario F skipped due to error: {exc}")

    if args.scenario in {"ALL", "G"}:
        try:
            from strategies.ml.feature_engineer import FeatureEngineer
            from strategies.ml.ml_strategy import MLStrategy
            from strategies.ml.model_registry import ModelRegistry
            from strategies.ml.signal_filter import SignalFilter

            registry = ModelRegistry(registry_path=str(registry_path))
            signal_filter = SignalFilter(
                min_confidence=float(ml_strategy_cfg.get("min_confidence", 0.45)),
                regime_filter=None,
                min_holding_days=int(ml_strategy_cfg.get("min_holding_days", 3)),
                signal_smoothing=bool(ml_strategy_cfg.get("signal_smoothing", True)),
            )
            ml_strategy = MLStrategy(
                symbol="BTCUSDT",
                model_registry=registry,
                feature_engineer=FeatureEngineer(asset_class="crypto"),
                signal_filter=signal_filter,
            )
            if "BTCUSDT" in data_f:
                holdout_btc = data_f["BTCUSDT"][data_f["BTCUSDT"].index >= holdout_start]
                if holdout_btc.empty:
                    raise RuntimeError(
                        f"No BTCUSDT holdout data available from {holdout_start.date().isoformat()} onward."
                    )
                scenario_g = run(
                    strategy=ml_strategy,
                    data={"BTCUSDT": holdout_btc},
                    initial_capital=initial_capital,
                    cost_model=crypto_cost_model,
                    rebalance_frequency="D",
                    sizing_method="equal_weight",
                    trading_hours="24/7",
                )
                results["G (ML)"] = ("crypto", scenario_g)
        except Exception as exc:
            print(f"\nWARNING: Scenario G (ML) skipped due to error: {exc}")

    print(
        f"\nML Split: train <= {train_end.date().isoformat()} | holdout >= {holdout_start.date().isoformat()}"
    )

    rows = [_format_result_row(label, result, asset_class) for label, (asset_class, result) in results.items()]
    print("\nSzenario-Vergleich:")
    _print_table(rows)

    if results:
        _plot_comparison({label: result for label, (_, result) in results.items()}, output_path="equity_curves_comparison.png")
        print("\nSaved: equity_curves_comparison.png")
    if run_equity_block:
        print("Saved: optimization_results.csv")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1) from exc

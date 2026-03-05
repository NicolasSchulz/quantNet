from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml


def load_settings() -> dict:
    with Path("config/settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from backtesting.cost_model import CostModel
    from backtesting.engine import run
    from data.ingestion.feed_factory import FeedFactory
    from data.normalizer import normalize_ohlcv
    from data.storage.parquet_store import ParquetStore
    from strategies.filters.regime_filter import RegimeFilter
    from strategies.ml.feature_engineer import FeatureEngineer
    from strategies.ml.ml_strategy import MLStrategy
    from strategies.ml.model_registry import ModelRegistry
    from strategies.ml.signal_filter import SignalFilter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("e2e_test")

    settings = load_settings()
    symbol = "SPY"
    feature_version = str(settings["ml"]["features"].get("feature_version", "v1"))
    end = pd.Timestamp.utcnow().date().isoformat()
    start = (pd.Timestamp.utcnow() - pd.DateOffset(years=2)).date().isoformat()

    logger.info("Step 1: build features")
    subprocess.run(
        [sys.executable, "feature_pipeline.py", "--symbol", symbol, "--start", start, "--end", end],
        check=True,
    )

    logger.info("Step 2: train model (walk-forward)")
    subprocess.run(
        [sys.executable, "model_training.py", "--symbol", symbol, "--feature-version", feature_version],
        check=True,
    )

    logger.info("Step 3: instantiate ML strategy from registry")
    registry = ModelRegistry(registry_path=str(settings["ml"]["registry"]["path"]))
    strategy = MLStrategy(
        symbol=symbol,
        model_registry=registry,
        feature_engineer=FeatureEngineer(),
        signal_filter=SignalFilter(
            min_confidence=float(settings["ml"]["strategy"]["min_confidence"]),
            regime_filter=RegimeFilter() if bool(settings["ml"]["strategy"].get("use_regime_filter", True)) else None,
            min_holding_days=int(settings["ml"]["strategy"]["min_holding_days"]),
            signal_smoothing=bool(settings["ml"]["strategy"].get("signal_smoothing", True)),
        ),
    )

    logger.info("Step 4: run backtest")
    interval = settings["data"]["intervals"]["primary"]
    store = ParquetStore(storage_path=settings["data"]["storage_path"])
    if store.exists(symbol, interval):
        ohlcv = store.load(symbol=symbol, interval=interval, start=start, end=end)
    else:
        feed = FeedFactory.create(config=settings)
        raw = feed.fetch_historical(symbol=symbol, start=start, end=end, interval=interval)
        ohlcv = normalize_ohlcv(raw, symbol=symbol, asset_class="etf", interval=interval)
        store.save(ohlcv, symbol=symbol, interval=interval)

    cost_cfg = settings["backtesting"]["cost_model"]
    cost_model = CostModel(
        commission=float(cost_cfg["commission"]),
        slippage_bps=float(cost_cfg["slippage_bps"]),
        spread_bps=float(cost_cfg["spread_bps"]),
    )
    result = run(
        strategy=strategy,
        data={symbol: ohlcv},
        initial_capital=float(settings["backtesting"]["initial_capital"]),
        cost_model=cost_model,
        rebalance_frequency="D",
        sizing_method="equal_weight",
    )

    logger.info("Step 5: assertions")
    if result.equity_curve.isna().any():
        raise RuntimeError("Equity curve contains NaN values")
    if len(result.trades) <= 0:
        raise RuntimeError("No trades executed in E2E test")

    signals = strategy.generate_signals(ohlcv)
    if not set(signals.unique()).issubset({-1, 0, 1}):
        raise RuntimeError("Generated signals contain invalid values")

    logger.info("E2E completed: trades=%d sharpe=%.3f", len(result.trades), result.metrics.get("sharpe_ratio", 0.0))

    # Persist quick smoke artifact
    out_dir = Path("outputs/ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, out_dir / "e2e_backtest_result.pkl")
    print("E2E smoke test passed")


if __name__ == "__main__":
    main()

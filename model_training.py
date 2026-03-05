from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from data.features.feature_store import FeatureStore
from data.ingestion.feed_factory import FeedFactory
from data.normalizer import normalize_ohlcv
from data.storage.parquet_store import ParquetStore
from strategies.ml.evaluator import ModelEvaluator
from strategies.ml.feature_engineer import FeatureEngineer
from strategies.ml.labeler import TripleBarrierLabeler
from strategies.ml.models.lgbm_classifier import LGBMClassifier
from strategies.ml.model_registry import ModelRegistry
from strategies.ml.walk_forward import WalkForwardConfig, WalkForwardValidator

LOGGER = logging.getLogger(__name__)


def load_settings(path: str = "config/settings.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_features(
    symbol: str,
    feature_version: str,
    settings: dict[str, Any],
    start: str,
    end: str,
) -> None:
    base_feature_path = settings["data"]["storage_path"] + "/features"
    store = FeatureStore(base_path=base_feature_path)
    if store.exists(symbol, feature_version):
        return

    LOGGER.info("Feature cache not found for %s/%s. Running feature_pipeline.py", symbol, feature_version)
    cmd = [
        sys.executable,
        "feature_pipeline.py",
        "--symbol",
        symbol,
        "--start",
        start,
        "--end",
        end,
    ]
    subprocess.run(cmd, check=True)


def _load_ohlcv(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: dict[str, Any]) -> pd.DataFrame:
    interval = settings["data"]["intervals"]["primary"]
    store = ParquetStore(storage_path=settings["data"]["storage_path"])

    if store.exists(symbol, interval):
        ohlcv = store.load(symbol=symbol, interval=interval, start=start, end=end)
        if not ohlcv.empty:
            return ohlcv

    feed = FeedFactory.create(config=settings)
    raw = feed.fetch_historical(symbol=symbol, start=start, end=end, interval=interval)
    ohlcv = normalize_ohlcv(raw, symbol=symbol, asset_class="etf", interval=interval)
    store.save(ohlcv, symbol=symbol, interval=interval)

    return ohlcv


def _load_ohlcv_returns(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: dict[str, Any]) -> pd.Series:
    ohlcv = _load_ohlcv(symbol=symbol, start=start, end=end, settings=settings)
    return ohlcv["close"].astype(float).pct_change().fillna(0.0)


def _run_hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    class_weights: dict[int, float] | None,
    base_params: dict[str, Any],
    config: WalkForwardConfig,
    output_dir: Path,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    evaluator = ModelEvaluator()

    param_grid = {
        "max_depth": [3, 4, 6],
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.01, 0.05],
        "min_child_samples": [30, 50, 100],
    }

    rows: list[dict[str, Any]] = []
    keys = list(param_grid.keys())
    for values in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(base_params)
        params.update(dict(zip(keys, values)))

        validator = WalkForwardValidator(
            model_class=LGBMClassifier,
            model_params=params,
            config=config,
        )
        wf_result = validator.run(X, y, class_weights=class_weights)
        trading = evaluator.compute_trading_metrics(
            y_pred=wf_result.predictions,
            returns=returns,
            transaction_cost_bps=transaction_cost_bps,
        )
        rows.append(
            {
                **dict(zip(keys, values)),
                "oos_sharpe": trading["strategy_sharpe"],
                "oos_cagr": trading["strategy_cagr"],
                "oos_max_dd": trading["strategy_max_drawdown"],
                "n_folds": len(wf_result.folds),
            }
        )

    out = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "hyperparameter_search.csv", index=False)
    LOGGER.info("Saved hyperparameter search to %s", output_dir / "hyperparameter_search.csv")
    if not out.empty:
        LOGGER.info("Best params by OOS Sharpe: %s", out.iloc[0].to_dict())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LGBM model with walk-forward validation")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = load_settings()

    symbol = args.symbol.upper()
    feature_version = args.feature_version

    lookback_years = int(settings["data"].get("lookback_years", 8))
    end = pd.Timestamp.utcnow()
    start = end - pd.DateOffset(years=lookback_years)

    _ensure_features(symbol, feature_version, settings, start.date().isoformat(), end.date().isoformat())

    feature_store = FeatureStore(base_path=settings["data"]["storage_path"] + "/features")
    X, y = feature_store.load(symbol=symbol, feature_version=feature_version)

    # Keep only samples with known labels.
    valid_idx = y.dropna().index
    X = X.reindex(valid_idx)
    y = y.reindex(valid_idx).astype(int)

    LOGGER.info("Loaded features: X=%s y=%s", X.shape, y.shape)

    labeling_cfg = settings["ml"]["labeling"]
    labeler = TripleBarrierLabeler(
        take_profit=float(labeling_cfg["take_profit"]),
        stop_loss=float(labeling_cfg["stop_loss"]),
        max_holding=int(labeling_cfg["max_holding"]),
        min_return=float(labeling_cfg["min_return"]),
        handle_imbalance=str(labeling_cfg.get("handle_imbalance", "weights")),
    )
    class_weights = labeler.get_class_weights(y)
    LOGGER.info("Class weights: %s", class_weights)

    train_cfg = settings["ml"]["training"]
    wf_config = WalkForwardConfig(
        train_window_days=int(train_cfg["train_window_days"]),
        test_window_days=int(train_cfg["test_window_days"]),
        step_size_days=int(train_cfg["step_size_days"]),
        purge_days=int(train_cfg["purge_days"]),
        embargo_days=int(train_cfg["embargo_days"]),
    )

    model_params = dict(settings["ml"]["model"].get("params", {}))
    validator = WalkForwardValidator(
        model_class=LGBMClassifier,
        model_params=model_params,
        config=wf_config,
    )

    wf_result = validator.run(X, y, class_weights=class_weights)

    returns = _load_ohlcv_returns(
        symbol=symbol,
        start=X.index.min(),
        end=X.index.max(),
        settings=settings,
    )

    evaluator = ModelEvaluator()
    cls_metrics = evaluator.compute_classification_metrics(
        y_true=y.reindex(wf_result.predictions.index),
        y_pred=wf_result.predictions.to_numpy(),
    )
    wf_result.aggregate_metrics.update(cls_metrics)

    trading_metrics = evaluator.compute_trading_metrics(
        y_pred=wf_result.predictions,
        returns=returns,
        transaction_cost_bps=float(train_cfg.get("transaction_cost_bps", 7.0)),
    )

    wf_result.metrics_per_fold = wf_result.metrics_per_fold.copy()
    if not wf_result.metrics_per_fold.empty:
        # attach fold-level strategy sharpe using fold prediction slices
        fold_sharpes: list[float] = []
        for _, row in wf_result.metrics_per_fold.iterrows():
            mask = (wf_result.predictions.index >= row["test_start"]) & (wf_result.predictions.index <= row["test_end"])
            fold_pred = wf_result.predictions.loc[mask]
            fold_ret = returns.reindex(fold_pred.index)
            fold_tm = evaluator.compute_trading_metrics(
                y_pred=fold_pred,
                returns=fold_ret,
                transaction_cost_bps=float(train_cfg.get("transaction_cost_bps", 7.0)),
            )
            fold_sharpes.append(float(fold_tm["strategy_sharpe"]))
        wf_result.metrics_per_fold["strategy_sharpe"] = fold_sharpes

    output_dir = Path(train_cfg.get("results_output_dir", "./outputs/ml/"))
    model_dir = Path(train_cfg.get("model_output_dir", "./models/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_results(wf_result, returns, output_dir=str(output_dir))
    report = evaluator.generate_report(wf_result, trading_metrics)
    print(report)

    # Final model on all data.
    final_params = dict(model_params)
    if class_weights:
        final_params["class_weights"] = class_weights
    final_model = LGBMClassifier(params=final_params)
    final_model.fit(X, y)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"lgbm_{feature_version}_{timestamp}.joblib"
    final_model.save(str(model_path))
    scaler_path = model_dir / f"lgbm_{feature_version}_{timestamp}.scaler.joblib"
    try:
        ohlcv_for_scaler = _load_ohlcv(
            symbol=symbol,
            start=X.index.min(),
            end=X.index.max(),
            settings=settings,
        )
        scaler_engineer = FeatureEngineer()
        _ = scaler_engineer.fit_transform(ohlcv_for_scaler)
        scaler_engineer.save_scaler(str(scaler_path))
        LOGGER.info("Saved scaler artifact: %s", scaler_path)
    except Exception as exc:  # pragma: no cover - runtime/data dependent
        LOGGER.warning("Could not persist scaler artifact: %s", exc)
    wf_path = output_dir / f"walk_forward_result_{symbol}_{feature_version}_{timestamp}.pkl"
    joblib.dump(wf_result, wf_path)
    LOGGER.info("Saved model: %s", model_path)

    registry = ModelRegistry(registry_path=str(settings["ml"]["registry"]["path"]))
    model_id = f"lgbm_{symbol}_{feature_version}_{timestamp}"
    registry.register(
        model_id=model_id,
        model_path=str(model_path),
        symbol=symbol,
        feature_version=feature_version,
        walk_forward_metrics=trading_metrics,
        trained_at=datetime.now(timezone.utc).isoformat(),
    )
    registry.set_default(symbol, model_id)
    LOGGER.info("Registered model '%s' and set as default for %s", model_id, symbol)

    if args.optimize:
        _ = _run_hyperparameter_search(
            X=X,
            y=y,
            returns=returns,
            class_weights=class_weights,
            base_params=model_params,
            config=wf_config,
            output_dir=output_dir,
            transaction_cost_bps=float(train_cfg.get("transaction_cost_bps", 7.0)),
        )


if __name__ == "__main__":
    main()

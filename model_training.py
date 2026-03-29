from __future__ import annotations

import argparse
import json
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
from strategies.ml.nested_walk_forward import NestedWalkForwardConfig, NestedWalkForwardValidator
from strategies.ml.models.lgbm_classifier import LGBMClassifier
from strategies.ml.model_registry import ModelRegistry
from strategies.ml.signal_filter import SignalFilter
from strategies.ml.training_progress import TrainingProgressTracker
from strategies.ml.walk_forward import WalkForwardConfig, WalkForwardValidator

LOGGER = logging.getLogger(__name__)


def load_settings(path: str = "config/settings.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_features(
    symbol: str,
    asset_class: str,
    feature_version: str,
    settings: dict[str, Any],
    start: str,
    end: str,
) -> None:
    base_feature_path = settings["data"]["storage_path"] + "/features"
    store = FeatureStore(base_path=base_feature_path)
    entry_strategy = str(settings.get("ml", {}).get("labeling", {}).get("entry_strategy", "all_candles")).lower()
    entry_strategy_config = settings.get("ml", {}).get("labeling", {}).get(entry_strategy, {})
    if store.exists(symbol, feature_version):
        metadata = store.metadata(symbol, feature_version)
        cached_strategy = str(metadata.get("entry_strategy", "all_candles")).lower()
        cached_strategy_config = metadata.get("entry_strategy_config", {})
        if cached_strategy == entry_strategy and cached_strategy_config == entry_strategy_config:
            return
        LOGGER.info(
            "Feature cache strategy mismatch for %s/%s: cached=%s, requested=%s. Rebuilding cache.",
            symbol,
            feature_version,
            cached_strategy,
            entry_strategy,
        )

    LOGGER.info("Feature cache not found for %s/%s. Running feature_pipeline.py", symbol, feature_version)
    cmd = [
        sys.executable,
        "feature_pipeline.py",
        "--symbol",
        symbol,
        "--asset-class",
        asset_class,
        "--start",
        start,
        "--end",
        end,
    ]
    subprocess.run(cmd, check=True)


def _load_ohlcv(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: dict[str, Any]) -> pd.DataFrame:
    interval = settings["data"]["intervals"]["primary"]
    store = ParquetStore(storage_path=settings["data"]["storage_path"])
    asset_class = "crypto" if symbol.upper().endswith("USDT") else "equity"
    start_str = pd.Timestamp(start).date().isoformat()
    end_str = pd.Timestamp(end).date().isoformat()

    if store.exists(symbol, interval):
        ohlcv = store.load(symbol=symbol, interval=interval, start=start, end=end)
        if not ohlcv.empty:
            return ohlcv

    feed = FeedFactory.create(config=settings, symbol=symbol)
    raw = feed.fetch_historical(symbol=symbol, start=start_str, end=end_str, interval=interval)
    ohlcv = normalize_ohlcv(raw, symbol=symbol, asset_class=asset_class, interval=interval)
    store.save(ohlcv, symbol=symbol, interval=interval)

    return ohlcv


def _load_ohlcv_returns(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: dict[str, Any]) -> pd.Series:
    ohlcv = _load_ohlcv(symbol=symbol, start=start, end=end, settings=settings)
    return ohlcv["close"].astype(float).pct_change().fillna(0.0)


def _split_final_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    val_fraction: float,
    val_purge_days: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    val_days = max(1, int(round(max(len(X), 1) * val_fraction)))
    val_start_idx = max(0, len(X) - val_days)
    val_start = X.index[val_start_idx]
    train_end = val_start - pd.Timedelta(days=val_purge_days)
    train_mask = X.index <= train_end
    val_mask = X.index >= val_start
    return X.loc[train_mask], y.loc[train_mask], X.loc[val_mask], y.loc[val_mask]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LGBM model with walk-forward validation")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--train-end", default=None, help="Inclusive train cutoff date (YYYY-MM-DD)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = load_settings()
    progress_path = str(settings["ml"]["training"].get("progress_path", "./outputs/ml/training_progress.json"))

    symbol = args.symbol.upper()
    asset_class = "crypto" if symbol.endswith("USDT") else "equity"
    feature_version = args.feature_version
    dataset_cfg = settings.get("ml", {}).get("dataset", {})
    train_end = pd.Timestamp(args.train_end or dataset_cfg.get("train_end", "2025-12-31"), tz="UTC")
    progress = TrainingProgressTracker(progress_path, symbol=symbol, feature_version=args.feature_version, optimize=args.optimize)

    lookback_years = int(settings["data"].get("lookback_years", 8))
    end = train_end
    start = end - pd.DateOffset(years=lookback_years)

    progress.update(phase="features", message="Pruefe/vervollstaendige Feature-Cache", percent=2.0)
    _ensure_features(symbol, asset_class, feature_version, settings, start.date().isoformat(), end.date().isoformat())

    feature_store = FeatureStore(base_path=settings["data"]["storage_path"] + "/features")
    progress.update(phase="loading_data", message="Lade Rohfeatures und Labels", percent=5.0)
    X, y = feature_store.load_raw(symbol=symbol, feature_version=feature_version, end=train_end)

    # Keep only samples with known labels.
    valid_idx = y.dropna().index
    X = X.reindex(valid_idx)
    y = y.reindex(valid_idx).astype(int)

    LOGGER.info("Loaded features: X=%s y=%s", X.shape, y.shape)
    LOGGER.info("Training cutoff for %s: %s", symbol, train_end.date().isoformat())

    labeling_cfg = settings["ml"]["labeling"][asset_class]
    labeler = TripleBarrierLabeler(
        take_profit=float(labeling_cfg["take_profit"]),
        stop_loss=float(labeling_cfg["stop_loss"]),
        max_holding=int(labeling_cfg["max_holding"]),
        min_return=float(labeling_cfg["min_return"]),
        handle_imbalance=str(labeling_cfg.get("handle_imbalance", "weights")),
        asset_class=asset_class,
    )
    class_weights = labeler.get_class_weights(y)
    LOGGER.info("Class weights: %s", class_weights)

    entry_strategy = str(settings.get("ml", {}).get("labeling", {}).get("entry_strategy", "all_candles")).lower()
    train_cfg = settings["ml"]["training"]
    wf_config = WalkForwardConfig(
        train_window_days=int(train_cfg["train_window_days"]),
        test_window_days=int(train_cfg["test_window_days"]),
        step_size_days=int(train_cfg["step_size_days"]),
        purge_days=int(train_cfg["purge_days"]),
        embargo_days=int(train_cfg["embargo_days"]),
        val_fraction=float(train_cfg.get("val_fraction", 0.15)),
        val_purge_days=int(train_cfg.get("val_purge_days", 5)),
        overfitting_threshold=float(train_cfg.get("overfitting_threshold", 0.5)),
    )

    model_params = dict(settings["ml"]["model"].get("params", {}))
    returns = _load_ohlcv_returns(
        symbol=symbol,
        start=X.index.min(),
        end=X.index.max(),
        settings=settings,
    )
    feature_engineer = FeatureEngineer(
        config={
            "feature_groups": settings["ml"]["features"]["crypto_groups"] if asset_class == "crypto" else settings["ml"]["features"]["groups"],
            "normalization": settings["ml"]["features"]["normalization"],
            "lookback_periods": settings["ml"]["features"].get("lookback_periods", {}),
            "warmup_bars": int(settings["ml"]["features"].get("lookback_periods", {}).get("warmup_bars", 200)),
            "mss_strategy": settings.get("ml", {}).get("labeling", {}).get("mss", {}),
        },
        asset_class=asset_class,
    )
    signal_cfg = settings["ml"].get("signal_filter", {})
    signal_strategy_cfg = signal_cfg.get(entry_strategy, {}) if isinstance(signal_cfg.get(entry_strategy), dict) else {}
    strategy_cfg = settings["ml"].get("strategy", {})
    strategy_entry_cfg = strategy_cfg.get(entry_strategy, {}) if isinstance(strategy_cfg.get(entry_strategy), dict) else {}
    min_confidence = float(signal_strategy_cfg.get("fallback_threshold", signal_cfg.get("fallback_threshold", strategy_cfg.get("min_confidence", 0.45))))
    min_holding_days = int(strategy_entry_cfg.get("min_holding_days", strategy_cfg.get("min_holding_days", 3)))
    signal_smoothing = bool(signal_strategy_cfg.get("signal_smoothing", signal_cfg.get("signal_smoothing", True)))
    signal_filter = SignalFilter(
        min_confidence=min_confidence,
        min_holding_days=min_holding_days,
        signal_smoothing=signal_smoothing,
    )
    threshold_optimization = bool(signal_strategy_cfg.get("threshold_optimization", signal_cfg.get("threshold_optimization", False)))
    threshold_candidates = [float(v) for v in signal_strategy_cfg.get("threshold_candidates", signal_cfg.get("threshold_candidates", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]))]
    fallback_threshold = float(signal_strategy_cfg.get("fallback_threshold", signal_cfg.get("fallback_threshold", strategy_cfg.get("min_confidence", 0.45))))
    min_threshold_optimization_samples = int(signal_strategy_cfg.get("min_val_samples_for_threshold_optimization", signal_cfg.get("min_val_samples_for_threshold_optimization", 0)))

    try:
        if args.optimize:
            nested_cfg = settings["ml"]["nested_walk_forward"]
            outer_cfg = WalkForwardConfig(**nested_cfg["outer"])
            inner_cfg = WalkForwardConfig(**nested_cfg["inner"])
            nested_validator = NestedWalkForwardValidator(
                model_class=LGBMClassifier,
            config=NestedWalkForwardConfig(outer=outer_cfg, inner=inner_cfg, param_grid=dict(nested_cfg["param_grid"])),
            feature_engineer=feature_engineer,
            signal_filter=signal_filter,
            transaction_cost_bps=float(train_cfg.get("transaction_cost_bps", 7.0)),
            threshold_optimization=threshold_optimization,
            threshold_candidates=threshold_candidates,
            fallback_threshold=fallback_threshold,
            )
            nested_result = nested_validator.run(
                X,
                y,
                returns=returns,
                class_weights=class_weights,
                base_params=model_params,
                progress_callback=lambda payload: progress.update(**payload),
            )
            wf_result = nested_result.walk_forward_result
            wf_result.params_stability = nested_result.params_stability
            model_params.update(
                {
                    key: (float(summary["most_common"]) if "." in str(summary["most_common"]) else int(summary["most_common"]))
                    for key, summary in nested_result.params_stability.items()
                    if summary.get("most_common") is not None
                }
            )
            LOGGER.info("Nested Walk-Forward params_stability: %s", nested_result.params_stability)
        else:
            progress.update(phase="walk_forward", message="Fuehre Walk-Forward Validierung aus", percent=35.0)
            validator = WalkForwardValidator(
                model_class=LGBMClassifier,
                model_params=model_params,
                config=wf_config,
                feature_engineer=feature_engineer,
                signal_filter=signal_filter,
                threshold_optimization=threshold_optimization,
                threshold_candidates=threshold_candidates,
                fallback_threshold=fallback_threshold,
                min_threshold_optimization_samples=min_threshold_optimization_samples,
                transaction_cost_bps=float(train_cfg.get("transaction_cost_bps", 7.0)),
            )
            wf_result = validator.run(X, y, returns=returns, class_weights=class_weights)

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

        output_dir = Path(train_cfg.get("results_output_dir", "./outputs/ml/"))
        model_dir = Path(train_cfg.get("model_output_dir", "./models/"))
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        progress.update(phase="reporting", message="Berechne Reports und Charts", percent=92.0)
        evaluator.plot_results(wf_result, returns, output_dir=str(output_dir))
        report = evaluator.generate_report(wf_result, trading_metrics)
        print(report)

        progress.update(phase="final_model", message="Trainiere finales Modell", percent=95.0)
        final_params = dict(model_params)
        if class_weights:
            final_params["class_weights"] = class_weights
        final_model = LGBMClassifier(params=final_params)
        final_scaler = feature_engineer.fit_scaler(X)
        X_scaled = feature_engineer.scale_features(X, final_scaler)
        X_train_final, y_train_final, X_val_final, y_val_final = _split_final_train_validation(
            X_scaled,
            y,
            val_fraction=float(train_cfg.get("val_fraction", 0.15)),
            val_purge_days=int(train_cfg.get("val_purge_days", 5)),
        )
        final_model.fit(X_train_final, y_train_final, X_val=X_val_final, y_val=y_val_final)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"lgbm_{feature_version}_{timestamp}.joblib"
        final_model.save(str(model_path))
        scaler_path = model_dir / f"lgbm_{feature_version}_{timestamp}.scaler.joblib"
        try:
            joblib.dump(final_scaler, scaler_path)
            LOGGER.info("Saved scaler artifact: %s", scaler_path)
        except Exception as exc:  # pragma: no cover - runtime/data dependent
            LOGGER.warning("Could not persist scaler artifact: %s", exc)
        wf_path = output_dir / f"walk_forward_result_{symbol}_{feature_version}_{timestamp}.pkl"
        joblib.dump(wf_result, wf_path)
        LOGGER.info("Saved model: %s", model_path)
        loss_curves_path = model_dir / f"lgbm_{symbol}_{feature_version}_{timestamp}_loss_curves.json"
        loss_payload = {
            "model_id": f"lgbm_{symbol}_{feature_version}_{timestamp}",
            "n_folds": len(wf_result.folds),
            "aggregate": wf_result.aggregate_loss,
            "folds": wf_result.loss_histories,
        }
        loss_curves_path.write_text(json.dumps(loss_payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved loss curves: %s", loss_curves_path)

        registry = ModelRegistry(registry_path=str(settings["ml"]["registry"]["path"]))
        model_id = f"lgbm_{symbol}_{feature_version}_{timestamp}"
        registry.register(
            model_id=model_id,
            model_path=str(model_path),
            symbol=symbol,
            feature_version=feature_version,
            walk_forward_metrics=trading_metrics,
            trained_at=datetime.now(timezone.utc).isoformat(),
            analysis={
                "stability": evaluator.compute_stability_risk(wf_result.metrics_per_fold),
                "overfitting": wf_result.overfitting_gap,
                "threshold_analysis": wf_result.threshold_stability,
                "params_stability": wf_result.params_stability,
            },
            loss_curves_path=str(loss_curves_path),
        )
        registry.set_default(symbol, model_id)
        LOGGER.info("Registered model '%s' and set as default for %s", model_id, symbol)
        progress.complete(model_id=model_id)
    except Exception as exc:
        progress.fail(str(exc))
        raise


if __name__ == "__main__":
    main()

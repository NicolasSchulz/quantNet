from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from data.storage.parquet_store import ParquetStore
from strategies.ml.evaluator import ModelEvaluator
from strategies.ml.model_registry import ModelRegistry


def _load_settings() -> dict:
    with Path("config/settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _registry_payload() -> dict:
    registry = ModelRegistry(registry_path=_load_settings()["ml"]["registry"]["path"])
    return registry._read()  # noqa: SLF001


def _model_info(model_id: str) -> Optional[dict[str, object]]:
    return _registry_payload().get("models", {}).get(model_id)


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    return datetime.fromisoformat(str(value)).astimezone(timezone.utc)


def _walk_forward_path(model_id: str, symbol: str, feature_version: str) -> Optional[Path]:
    suffix = f"{symbol.upper()}_{feature_version}_"
    if suffix in model_id:
        timestamp = model_id.split(suffix, maxsplit=1)[1]
        path = Path("outputs/ml") / f"walk_forward_result_{symbol.upper()}_{feature_version}_{timestamp}.pkl"
        if path.exists():
            return path
    candidates = sorted(Path("outputs/ml").glob(f"walk_forward_result_{symbol.upper()}_{feature_version}_*.pkl"))
    return candidates[-1] if candidates else None


def _load_walk_forward_result(model_id: str, symbol: str, feature_version: str):
    path = _walk_forward_path(model_id, symbol, feature_version)
    if path is None or not path.exists():
        return None
    return joblib.load(path)


def _precision_recall_from_confusion(cm: np.ndarray, label_index: int) -> tuple[float, float]:
    tp = float(cm[label_index, label_index])
    precision_den = float(cm[:, label_index].sum())
    recall_den = float(cm[label_index, :].sum())
    precision = tp / precision_den if precision_den > 0 else 0.0
    recall = tp / recall_den if recall_den > 0 else 0.0
    return precision, recall


def _load_returns(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    settings = _load_settings()
    interval = settings["data"]["intervals"]["primary"]
    store = ParquetStore(storage_path=settings["data"]["storage_path"])
    ohlcv = store.load(symbol=symbol, interval=interval, start=start, end=end)
    return ohlcv["close"].astype(float).pct_change().fillna(0.0)


def _normalized_confusion(wf_result) -> np.ndarray:
    aggregate = dict(getattr(wf_result, "aggregate_metrics", {}))
    matrix = np.asarray(aggregate.get("confusion_matrix", np.zeros((3, 3))), dtype=int)
    normalized = []
    for row in matrix:
        total = int(row.sum()) or 1
        normalized.append([round(float(value) / total, 4) for value in row])
    return np.asarray(normalized, dtype=float)


def _stability_and_gap(wf_result) -> tuple[dict[str, object], dict[str, object]]:
    evaluator = ModelEvaluator()
    metrics_per_fold = getattr(wf_result, "metrics_per_fold", pd.DataFrame())
    train_metrics_per_fold = getattr(wf_result, "train_metrics_per_fold", pd.DataFrame())
    stability = evaluator.compute_stability_risk(metrics_per_fold)
    overfitting = evaluator.compute_overfitting_gap(train_metrics_per_fold, metrics_per_fold)
    return stability, overfitting


def _threshold_analysis(wf_result) -> dict[str, object]:
    evaluator = ModelEvaluator()
    threshold_per_fold = getattr(wf_result, "threshold_per_fold", pd.Series(dtype=float))
    source = "fixed"
    if not threshold_per_fold.empty and hasattr(wf_result, "metrics_per_fold") and not wf_result.metrics_per_fold.empty:
        source = str(wf_result.metrics_per_fold.get("threshold_source", pd.Series(["fixed"])).iloc[0])
    return evaluator.compute_threshold_stability(threshold_per_fold, source=source)


def _build_fold_rows(wf_result) -> list[dict[str, object]]:
    metrics_per_fold = getattr(wf_result, "metrics_per_fold", pd.DataFrame()).copy()
    train_metrics_per_fold = getattr(wf_result, "train_metrics_per_fold", pd.DataFrame()).copy()
    if metrics_per_fold.empty:
        return []
    if not train_metrics_per_fold.empty:
        metrics_per_fold = metrics_per_fold.merge(train_metrics_per_fold, on="fold_id", how="left")

    rows: list[dict[str, object]] = []
    for _, row in metrics_per_fold.iterrows():
        rows.append(
            {
                "fold_id": int(row["fold_id"]),
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "val_start": row.get("val_start", row["train_end"]),
                "val_end": row.get("val_end", row["train_end"]),
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "train_accuracy": float(row.get("train_accuracy", 0.0)),
                "train_sharpe": float(row.get("train_sharpe", 0.0)),
                "train_f1_macro": float(row.get("train_f1_macro", 0.0)),
                "oos_accuracy": float(row.get("oos_accuracy", row.get("accuracy", 0.0))),
                "oos_sharpe": float(row.get("oos_sharpe", row.get("strategy_sharpe", row.get("fold_sharpe", 0.0)))),
                "oos_f1_macro": float(row.get("oos_f1_macro", row.get("f1_macro", 0.0))),
                "accuracy_gap": float(row.get("accuracy_gap", 0.0)),
                "sharpe_gap": float(row.get("sharpe_gap", 0.0)),
                "overfitting_flag": bool(row.get("overfitting_flag", False)),
                "optimal_threshold": float(row.get("optimal_threshold", 0.45)),
                "threshold_source": str(row.get("threshold_source", "fixed")),
                "n_trades": int(row.get("n_test_samples", 0)),
                "cagr": 0.0,
            }
        )
    return rows


def list_models() -> dict[str, list[dict[str, object]]]:
    payload = _registry_payload()
    models: list[dict[str, object]] = []
    for model_id, info in payload.get("models", {}).items():
        models.append(
            {
                "model_id": model_id,
                "symbol": str(info.get("symbol", "")),
                "trained_at": _parse_timestamp(info.get("trained_at")),
                "sharpe": float(info.get("metrics", {}).get("sharpe", 0.0)),
                "status": str(info.get("status", "active")),
                "feature_version": str(info.get("feature_version", "v1")),
            }
        )
    models.sort(key=lambda row: row["trained_at"], reverse=True)
    return {"models": models}


def get_model_metrics(model_id: str) -> Optional[dict[str, object]]:
    info = _model_info(model_id)
    if info is None:
        return None

    symbol = str(info.get("symbol", "")).upper()
    feature_version = str(info.get("feature_version", "v1"))
    trained_at = _parse_timestamp(info.get("trained_at"))
    wf_result = _load_walk_forward_result(model_id, symbol, feature_version)
    if wf_result is None:
        return None

    aggregate = dict(getattr(wf_result, "aggregate_metrics", {}))
    confusion = np.asarray(aggregate.get("confusion_matrix", np.zeros((3, 3))), dtype=float)
    short_precision, short_recall = _precision_recall_from_confusion(confusion, 0)

    stability, overfitting = _stability_and_gap(wf_result)
    threshold_analysis = _threshold_analysis(wf_result)

    trading_metrics = {
        "strategy_sharpe": float(info.get("metrics", {}).get("sharpe", 0.0)),
        "strategy_cagr": float(info.get("metrics", {}).get("cagr", 0.0)),
        "strategy_max_drawdown": float(info.get("metrics", {}).get("max_drawdown", 0.0)),
        "strategy_calmar": 0.0,
        "n_trades": 0,
        "win_rate": 0.0,
    }
    trading_metrics["strategy_calmar"] = float(
        trading_metrics["strategy_cagr"] / abs(trading_metrics["strategy_max_drawdown"])
    ) if trading_metrics["strategy_max_drawdown"] not in {0.0, -0.0} else 0.0

    if not getattr(wf_result, "predictions", pd.Series(dtype=float)).empty:
        try:
            returns = _load_returns(symbol, wf_result.predictions.index.min(), wf_result.predictions.index.max())
            evaluator = ModelEvaluator()
            trading_metrics.update(
                evaluator.compute_trading_metrics(
                    y_pred=wf_result.predictions,
                    returns=returns.reindex(wf_result.predictions.index).fillna(0.0),
                    transaction_cost_bps=float(_load_settings()["ml"]["training"].get("transaction_cost_bps", 7.0)),
                )
            )
        except Exception:
            pass

    return {
        "model_id": model_id,
        "symbol": symbol,
        "trained_at": trained_at,
        "feature_version": feature_version,
        "n_folds": len(getattr(wf_result, "folds", [])),
        "train_window_days": int(aggregate.get("train_window_days", 0)),
        "test_window_days": int(aggregate.get("test_window_days", 0)),
        "accuracy": float(aggregate.get("accuracy", 0.0)),
        "f1_macro": float(aggregate.get("f1_macro", 0.0)),
        "f1_long": float(aggregate.get("f1_long", 0.0)),
        "f1_short": float(aggregate.get("f1_short", 0.0)),
        "precision_long": float(aggregate.get("precision_long", 0.0)),
        "recall_long": float(aggregate.get("recall_long", 0.0)),
        "precision_short": short_precision,
        "recall_short": short_recall,
        "oos_sharpe": float(trading_metrics.get("strategy_sharpe", 0.0)),
        "oos_cagr": float(trading_metrics.get("strategy_cagr", 0.0)),
        "oos_max_drawdown": float(trading_metrics.get("strategy_max_drawdown", 0.0)),
        "oos_calmar": float(trading_metrics.get("strategy_calmar", 0.0)),
        "n_trades": int(trading_metrics.get("n_trades", 0)),
        "win_rate": float(trading_metrics.get("win_rate", 0.0)),
        "stability": stability,
        "overfitting": overfitting,
        "threshold_analysis": threshold_analysis,
        "roc_auc_long": None,
        "roc_auc_short": None,
    }


def get_default_model(symbol: str) -> Optional[dict[str, object]]:
    payload = _registry_payload()
    model_id = payload.get("default", {}).get(symbol.upper())
    if model_id is None:
        return None
    return get_model_metrics(str(model_id))


def get_confusion_matrix(model_id: str) -> Optional[dict[str, object]]:
    info = _model_info(model_id)
    if info is None:
        return None
    wf_result = _load_walk_forward_result(model_id, str(info.get("symbol", "")), str(info.get("feature_version", "v1")))
    if wf_result is None:
        return None
    matrix = np.asarray(getattr(wf_result, "aggregate_metrics", {}).get("confusion_matrix", np.zeros((3, 3))), dtype=int)
    normalized = _normalized_confusion(wf_result).tolist()
    return {"matrix": matrix.tolist(), "labels": ["Short", "Flat", "Long"], "normalized": normalized}


def get_fold_results(model_id: str) -> Optional[dict[str, list[dict[str, object]]]]:
    info = _model_info(model_id)
    if info is None:
        return None
    wf_result = _load_walk_forward_result(model_id, str(info.get("symbol", "")), str(info.get("feature_version", "v1")))
    if wf_result is None:
        return None
    return {"folds": _build_fold_rows(wf_result)}


def get_train_oos_comparison(model_id: str) -> Optional[dict[str, object]]:
    folds_payload = get_fold_results(model_id)
    metrics_payload = get_model_metrics(model_id)
    if folds_payload is None or metrics_payload is None:
        return None
    folds = folds_payload["folds"]
    return {
        "folds": folds,
        "summary": metrics_payload["overfitting"],
        "chart_data": {
            "fold_labels": [f"Fold {row['fold_id']}" for row in folds],
            "train_sharpes": [float(row["train_sharpe"]) for row in folds],
            "oos_sharpes": [float(row["oos_sharpe"]) for row in folds],
            "gaps": [float(row["sharpe_gap"]) for row in folds],
            "overfitting_flags": [bool(row["overfitting_flag"]) for row in folds],
        },
    }


def get_threshold_analysis(model_id: str) -> Optional[dict[str, object]]:
    payload = get_model_metrics(model_id)
    if payload is None:
        return None
    return payload["threshold_analysis"]


def get_feature_importance(model_id: str) -> Optional[dict[str, list[dict[str, object]]]]:
    info = _model_info(model_id)
    if info is None:
        return None
    wf_result = _load_walk_forward_result(model_id, str(info.get("symbol", "")), str(info.get("feature_version", "v1")))
    if wf_result is None or getattr(wf_result, "feature_importance_mean", pd.Series(dtype=float)).empty:
        return {"features": []}
    std = wf_result.feature_importance_std if wf_result.feature_importance_std is not None else pd.Series(dtype=float)
    rows = []
    for feature, importance in wf_result.feature_importance_mean.head(20).items():
        rows.append({"feature": str(feature), "importance": float(importance), "std": float(std.get(feature, 0.0))})
    return {"features": rows}


def get_loss_curves(model_id: str) -> Optional[dict[str, object]]:
    info = _model_info(model_id)
    if info is None:
        return None
    loss_path_value = info.get("loss_curves_path")
    if not loss_path_value:
        return None
    loss_path = Path(str(loss_path_value))
    if not loss_path.exists():
        return None

    payload = json.loads(loss_path.read_text(encoding="utf-8"))
    folds = []
    for fold in payload.get("folds", []):
        train = [float(v) for v in fold.get("train", [])]
        validation = [float(v) for v in fold.get("validation", [])]
        min_val_loss = min(validation) if validation else 0.0
        min_val_iteration = int(validation.index(min_val_loss)) if validation else 0
        folds.append(
            {
                "fold_id": int(fold.get("fold_id", 0)),
                "train_start": str(fold.get("train_start", "")),
                "train_end": str(fold.get("train_end", "")),
                "test_start": str(fold.get("test_start", "")),
                "test_end": str(fold.get("test_end", "")),
                "metric": str(fold.get("metric", "unknown")),
                "train": train,
                "validation": validation,
                "best_iteration": int(fold.get("best_iteration", 0)),
                "n_iterations": int(fold.get("n_iterations", len(train))),
                "early_stopped": bool(fold.get("early_stopped", False)),
                "final_train_loss": float(train[-1]) if train else 0.0,
                "final_val_loss": float(validation[-1]) if validation else 0.0,
                "min_val_loss": float(min_val_loss),
                "min_val_iteration": int(min_val_iteration),
                "overfit_gap": float((validation[-1] - train[-1]) if train and validation else 0.0),
            }
        )

    return {
        "model_id": str(payload.get("model_id", model_id)),
        "n_folds": int(payload.get("n_folds", len(folds))),
        "aggregate": payload.get("aggregate") or {
            "metric": "unknown",
            "train_mean": [],
            "train_std": [],
            "val_mean": [],
            "val_std": [],
            "min_iterations": 0,
            "max_iterations": 0,
            "common_iterations": 0,
            "n_folds_included": 0,
        },
        "folds": folds,
    }


def get_training_progress() -> dict[str, object]:
    settings = _load_settings()
    progress_path = Path(str(settings["ml"]["training"].get("progress_path", "./outputs/ml/training_progress.json")))
    if not progress_path.exists():
        return {
            "status": "idle",
            "symbol": None,
            "feature_version": None,
            "optimize": False,
            "phase": "idle",
            "message": "Kein Training aktiv",
            "percent": 0.0,
            "updated_at": datetime.now(timezone.utc),
            "outer_fold": None,
            "inner_combo_index": None,
            "inner_combo_total": None,
            "completed_steps": None,
            "total_steps": None,
            "model_id": None,
        }
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["updated_at"] = _parse_timestamp(payload.get("updated_at"))
    return payload

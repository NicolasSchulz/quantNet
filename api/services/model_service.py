from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

try:
    from api.schemas.model import OverfittingRisk
except ImportError:
    from schemas.model import OverfittingRisk


def is_live_mode() -> bool:
    return os.getenv("LIVE_MODE", "false").lower() == "true"


def _risk_from_stats(sharpe_cv: float, pct_profitable_folds: float) -> OverfittingRisk:
    if sharpe_cv > 1.0 or pct_profitable_folds < 0.5:
        return "HIGH"
    if sharpe_cv > 0.5 or pct_profitable_folds < 0.7:
        return "MEDIUM"
    return "LOW"


@lru_cache(maxsize=1)
def _mock_models() -> list[dict[str, object]]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return [
        {
            "model_id": "lgbm_SPY_v1_20240101",
            "symbol": "SPY",
            "trained_at": now - timedelta(days=45),
            "sharpe": 1.24,
            "status": "active",
            "feature_version": "v1",
        },
        {
            "model_id": "lgbm_QQQ_v2_20240214",
            "symbol": "QQQ",
            "trained_at": now - timedelta(days=18),
            "sharpe": 0.96,
            "status": "paper",
            "feature_version": "v2",
        },
    ]


def list_models() -> dict[str, list[dict[str, object]]]:
    if is_live_mode():
        try:
            from strategies.ml.model_registry import ModelRegistry

            frame = ModelRegistry().list_models()
            if not frame.empty:
                models = []
                for _, row in frame.iterrows():
                    models.append(
                        {
                            "model_id": row["model_id"],
                            "symbol": row["symbol"],
                            "trained_at": datetime.fromisoformat(str(row["trained_at"])).astimezone(timezone.utc)
                            if isinstance(row["trained_at"], str)
                            else row["trained_at"],
                            "sharpe": float(row["sharpe"]),
                            "status": row["status"],
                        }
                    )
                return {"models": models}
        except Exception:
            pass
    return {"models": _mock_models()}


def _build_metrics(model_id: str, symbol: str, trained_at: datetime, feature_version: str) -> dict[str, object]:
    n_folds = 18
    sharpe_mean = 0.92 if symbol == "SPY" else 0.74
    sharpe_std = 0.31 if symbol == "SPY" else 0.48
    sharpe_cv = sharpe_std / sharpe_mean
    pct_profitable_folds = 0.78 if symbol == "SPY" else 0.61
    return {
        "model_id": model_id,
        "symbol": symbol,
        "trained_at": trained_at,
        "feature_version": feature_version,
        "n_folds": n_folds,
        "train_window_days": 252,
        "test_window_days": 63,
        "accuracy": 0.523 if symbol == "SPY" else 0.497,
        "f1_macro": 0.504 if symbol == "SPY" else 0.462,
        "f1_long": 0.541 if symbol == "SPY" else 0.488,
        "f1_short": 0.471 if symbol == "SPY" else 0.429,
        "precision_long": 0.556 if symbol == "SPY" else 0.501,
        "recall_long": 0.528 if symbol == "SPY" else 0.477,
        "precision_short": 0.492 if symbol == "SPY" else 0.438,
        "recall_short": 0.451 if symbol == "SPY" else 0.421,
        "oos_sharpe": 1.18 if symbol == "SPY" else 0.83,
        "oos_cagr": 0.163 if symbol == "SPY" else 0.108,
        "oos_max_drawdown": -0.092 if symbol == "SPY" else -0.128,
        "oos_calmar": 1.77 if symbol == "SPY" else 0.84,
        "n_trades": 284 if symbol == "SPY" else 198,
        "win_rate": 0.548 if symbol == "SPY" else 0.519,
        "sharpe_mean": round(sharpe_mean, 3),
        "sharpe_std": round(sharpe_std, 3),
        "sharpe_cv": round(sharpe_cv, 3),
        "pct_profitable_folds": round(pct_profitable_folds, 3),
        "overfitting_risk": _risk_from_stats(sharpe_cv, pct_profitable_folds),
        "roc_auc_long": 0.612 if symbol == "SPY" else 0.571,
        "roc_auc_short": 0.584 if symbol == "SPY" else 0.549,
    }


def get_model_metrics(model_id: str) -> Optional[dict[str, object]]:
    for model in list_models()["models"]:
        if model["model_id"] == model_id:
            return _build_metrics(
                model_id=model_id,
                symbol=str(model["symbol"]),
                trained_at=model["trained_at"],
                feature_version=str(model.get("feature_version", "v1")),
            )
    return None


def get_default_model(symbol: str) -> Optional[dict[str, object]]:
    models = [model for model in list_models()["models"] if str(model["symbol"]).upper() == symbol.upper()]
    if not models:
        return None
    selected = max(models, key=lambda model: float(model["sharpe"]))
    return get_model_metrics(str(selected["model_id"]))


def get_confusion_matrix(model_id: str) -> Optional[dict[str, object]]:
    metrics = get_model_metrics(model_id)
    if metrics is None:
        return None
    matrix = [
        [58, 18, 11],
        [22, 91, 24],
        [13, 17, 64],
    ]
    normalized = []
    for row in matrix:
        total = sum(row) or 1
        normalized.append([round(value / total, 4) for value in row])
    return {"matrix": matrix, "labels": ["Short", "Flat", "Long"], "normalized": normalized}


def get_fold_results(model_id: str) -> Optional[dict[str, list[dict[str, object]]]]:
    metrics = get_model_metrics(model_id)
    if metrics is None:
        return None
    trained_at = metrics["trained_at"]
    folds = []
    for idx in range(metrics["n_folds"]):
        test_end = trained_at - timedelta(days=(metrics["n_folds"] - idx - 1) * metrics["test_window_days"])
        test_start = test_end - timedelta(days=metrics["test_window_days"])
        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=metrics["train_window_days"])
        sharpe = round(math.sin(idx / 2.8) * 0.9 + 0.55, 3)
        folds.append(
            {
                "fold_id": idx + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "sharpe": sharpe,
                "accuracy": round(0.47 + idx * 0.004 + (0.015 if sharpe > 0 else -0.01), 3),
                "n_trades": 8 + (idx % 7) * 3,
                "cagr": round(0.02 + sharpe * 0.08, 3),
            }
        )
    return {"folds": folds}


def get_feature_importance(model_id: str) -> Optional[dict[str, list[dict[str, object]]]]:
    if get_model_metrics(model_id) is None:
        return None
    features = [
        ("trend_sma_20_gap", 0.182, 0.024),
        ("trend_ema_50_gap", 0.164, 0.019),
        ("momentum_rsi_14", 0.142, 0.031),
        ("momentum_return_5d", 0.131, 0.021),
        ("volatility_atr_14", 0.119, 0.028),
        ("volume_obv_slope", 0.108, 0.014),
        ("trend_macd_hist", 0.099, 0.019),
        ("momentum_stoch_k", 0.093, 0.026),
        ("volatility_realized_10d", 0.087, 0.013),
        ("volume_zscore_20", 0.076, 0.018),
        ("trend_high_breakout", 0.072, 0.012),
        ("momentum_return_20d", 0.069, 0.022),
        ("volatility_range_compress", 0.061, 0.011),
        ("volume_dollar_turnover", 0.055, 0.017),
        ("trend_regime_score", 0.049, 0.010),
    ]
    return {
        "features": [{"feature": feature, "importance": importance, "std": std} for feature, importance, std in features]
    }

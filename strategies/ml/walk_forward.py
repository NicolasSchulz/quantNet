from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Type

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from strategies.ml.models.base_model import BaseModel
from strategies.ml.signal_filter import SignalFilter

if TYPE_CHECKING:
    from strategies.ml.feature_engineer import FeatureEngineer

LOGGER = logging.getLogger(__name__)


class _FoldScalerHelper:
    def __init__(self) -> None:
        self.normalization = "robust"

    def fit_scaler(self, X: pd.DataFrame) -> RobustScaler:
        scale_cols = [c for c in X.columns if c not in {"rsi_14", "rsi_28", "stoch_k", "stoch_d", "bb_position", "body_size", "upper_wick", "lower_wick"}]
        scaler = RobustScaler()
        if scale_cols:
            scaler.fit(X[scale_cols])
        return scaler

    def scale_features(self, X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
        scale_cols = [c for c in X.columns if c not in {"rsi_14", "rsi_28", "stoch_k", "stoch_d", "bb_position", "body_size", "upper_wick", "lower_wick"}]
        result = X.copy()
        if scale_cols:
            result[scale_cols] = scaler.transform(result[scale_cols])
        return result.astype("float64")


@dataclass
class WalkForwardConfig:
    train_window_days: int = 504
    test_window_days: int = 63
    step_size_days: int = 21
    purge_days: int = 5
    embargo_days: int = 10
    use_days: bool = True
    val_fraction: float = 0.15
    val_purge_days: int = 5
    overfitting_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.use_days:
            self.train_window_samples = self.train_window_days
            self.test_window_samples = self.test_window_days
            self.step_size_samples = self.step_size_days
            self.purge_samples = self.purge_days
            self.embargo_samples = self.embargo_days


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_samples: int
    n_val_samples: int
    n_test_samples: int
    scaler: RobustScaler | None = None
    train_feature_stats: dict[str, pd.Series] | None = None
    oos_accuracy: float = 0.0
    oos_sharpe: float = 0.0
    oos_f1_macro: float = 0.0
    train_accuracy: float = 0.0
    train_sharpe: float = 0.0
    train_f1_macro: float = 0.0
    accuracy_gap: float = 0.0
    sharpe_gap: float = 0.0
    overfitting_flag: bool = False
    optimal_threshold: float = 0.45
    threshold_source: str = "fixed"
    loss_history: dict[str, Any] | None = None


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    predictions: pd.Series
    probabilities: pd.DataFrame
    metrics_per_fold: pd.DataFrame
    train_metrics_per_fold: pd.DataFrame
    aggregate_metrics: dict[str, Any]
    feature_importance_mean: pd.Series
    feature_importance_std: pd.Series | None = None
    stability_risk: str = "HIGH"
    overfitting_gap: dict[str, Any] = field(default_factory=dict)
    threshold_per_fold: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    threshold_stability: dict[str, Any] = field(default_factory=dict)
    params_stability: dict[str, Any] = field(default_factory=dict)
    loss_histories: list[dict[str, Any]] = field(default_factory=list)
    aggregate_loss: dict[str, Any] | None = None


class WalkForwardValidator:
    """Walk-forward validator with purge, embargo, validation split, and fold-local scaling."""

    def __init__(
        self,
        model_class: Type[BaseModel],
        model_params: dict[str, Any],
        config: WalkForwardConfig,
        feature_engineer: Any | None = None,
        signal_filter: SignalFilter | None = None,
        threshold_optimization: bool = False,
        threshold_candidates: list[float] | None = None,
        fallback_threshold: float = 0.45,
        min_threshold_optimization_samples: int = 0,
        transaction_cost_bps: float = 7.0,
    ) -> None:
        self.model_class = model_class
        self.model_params = model_params
        self.config = config
        self.feature_engineer = feature_engineer or _FoldScalerHelper()
        self.signal_filter = signal_filter
        self.threshold_optimization = threshold_optimization
        self.threshold_candidates = threshold_candidates or [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        self.fallback_threshold = float(fallback_threshold)
        self.min_threshold_optimization_samples = int(min_threshold_optimization_samples)
        self.transaction_cost_bps = float(transaction_cost_bps)

    def _compute_aggregate_loss(self, loss_histories: list[dict[str, Any]]) -> dict[str, Any] | None:
        valid = [history for history in loss_histories if history.get("train") and history.get("validation")]
        if not valid:
            return None
        min_iters = min(len(history["train"]) for history in valid)
        max_iters = max(len(history["train"]) for history in valid)
        train_matrix = np.array([history["train"][:min_iters] for history in valid], dtype=float)
        val_matrix = np.array([history["validation"][:min_iters] for history in valid], dtype=float)
        return {
            "metric": str(valid[0].get("metric", "unknown")),
            "train_mean": train_matrix.mean(axis=0).tolist(),
            "train_std": train_matrix.std(axis=0).tolist(),
            "val_mean": val_matrix.mean(axis=0).tolist(),
            "val_std": val_matrix.std(axis=0).tolist(),
            "min_iterations": int(min_iters),
            "max_iterations": int(max_iters),
            "common_iterations": int(min_iters),
            "n_folds_included": int(len(valid)),
        }

    def generate_folds(self, index: pd.DatetimeIndex) -> list[WalkForwardFold]:
        if len(index) == 0:
            return []

        idx = pd.DatetimeIndex(index).sort_values().unique()
        start_cursor = idx[0]
        end_limit = idx[-1]

        folds: list[WalkForwardFold] = []
        fold_id = 1

        if self.config.use_days:
            train_td = pd.Timedelta(days=self.config.train_window_days)
            test_td = pd.Timedelta(days=self.config.test_window_days)
            step_td = pd.Timedelta(days=self.config.step_size_days)
            purge_td = pd.Timedelta(days=self.config.purge_days)
            embargo_td = pd.Timedelta(days=self.config.embargo_days)
            val_purge_td = pd.Timedelta(days=self.config.val_purge_days)
            val_days = max(1, int(round(self.config.train_window_days * self.config.val_fraction)))
            val_td = pd.Timedelta(days=val_days)

            while True:
                full_train_start = start_cursor
                full_train_end = full_train_start + train_td
                val_end = full_train_end
                val_start = val_end - val_td + pd.Timedelta(days=1)
                train_end = val_start - val_purge_td
                test_start = val_end + purge_td
                test_end = test_start + test_td

                if test_end > end_limit or train_end <= full_train_start:
                    break

                n_train = int(((idx >= full_train_start) & (idx <= train_end)).sum())
                n_val = int(((idx >= val_start) & (idx <= val_end)).sum())
                n_test = int(((idx >= test_start) & (idx <= test_end)).sum())

                if n_train > 0 and n_val > 0 and n_test > 0:
                    folds.append(
                        WalkForwardFold(
                            fold_id=fold_id,
                            train_start=full_train_start,
                            train_end=train_end,
                            val_start=val_start,
                            val_end=val_end,
                            test_start=test_start,
                            test_end=test_end,
                            n_train_samples=n_train,
                            n_val_samples=n_val,
                            n_test_samples=n_test,
                        )
                    )
                    fold_id += 1

                next_cursor = start_cursor + step_td
                min_cursor_for_embargo = test_end + embargo_td - train_td - purge_td
                if min_cursor_for_embargo > next_cursor:
                    next_cursor = min_cursor_for_embargo
                if next_cursor >= end_limit:
                    break
                start_cursor = next_cursor
        else:
            n = len(idx)
            start_idx = 0
            val_samples = max(1, int(round(self.config.train_window_days * self.config.val_fraction)))
            while True:
                full_train_start_idx = start_idx
                full_train_end_idx = full_train_start_idx + self.config.train_window_days - 1
                val_end_idx = full_train_end_idx
                val_start_idx = max(full_train_start_idx, val_end_idx - val_samples + 1)
                train_end_idx = val_start_idx - self.config.val_purge_days - 1
                test_start_idx = val_end_idx + self.config.purge_days + 1
                test_end_idx = test_start_idx + self.config.test_window_days - 1

                if test_end_idx >= n or train_end_idx <= full_train_start_idx:
                    break

                folds.append(
                    WalkForwardFold(
                        fold_id=fold_id,
                        train_start=idx[full_train_start_idx],
                        train_end=idx[train_end_idx],
                        val_start=idx[val_start_idx],
                        val_end=idx[val_end_idx],
                        test_start=idx[test_start_idx],
                        test_end=idx[test_end_idx],
                        n_train_samples=train_end_idx - full_train_start_idx + 1,
                        n_val_samples=val_end_idx - val_start_idx + 1,
                        n_test_samples=test_end_idx - test_start_idx + 1,
                    )
                )
                fold_id += 1

                next_idx = start_idx + self.config.step_size_days
                min_idx_for_embargo = test_end_idx + self.config.embargo_days - self.config.train_window_days - self.config.purge_days
                if min_idx_for_embargo > next_idx:
                    next_idx = min_idx_for_embargo
                if next_idx >= n:
                    break
                start_idx = next_idx

        LOGGER.info("Generated %d walk-forward folds", len(folds))
        return folds

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: pd.Series | None = None,
        class_weights: dict[int, float] | None = None,
    ) -> WalkForwardResult:
        from strategies.ml.evaluator import ModelEvaluator

        data_index = X.index.intersection(y.dropna().index)
        X_raw = X.loc[data_index].sort_index()
        y = y.loc[data_index].sort_index().astype(int)
        returns = returns.reindex(data_index).fillna(0.0).astype(float) if returns is not None else pd.Series(0.0, index=data_index)

        folds = self.generate_folds(X_raw.index)
        evaluator = ModelEvaluator()

        pred_chunks: list[pd.Series] = []
        proba_chunks: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, Any]] = []
        train_fold_metrics: list[dict[str, Any]] = []
        feature_importances: list[pd.Series] = []
        thresholds: list[dict[str, Any]] = []
        loss_histories: list[dict[str, Any]] = []

        for i, fold in enumerate(folds, start=1):
            train_mask = (X_raw.index >= fold.train_start) & (X_raw.index <= fold.train_end)
            val_mask = (X_raw.index >= fold.val_start) & (X_raw.index <= fold.val_end)
            test_mask = (X_raw.index >= fold.test_start) & (X_raw.index <= fold.test_end)

            X_train_raw = X_raw.loc[train_mask]
            y_train = y.loc[train_mask]
            X_val_raw = X_raw.loc[val_mask]
            y_val = y.loc[val_mask]
            X_test_raw = X_raw.loc[test_mask]
            y_test = y.loc[test_mask]

            if len(X_train_raw) < 20 or len(X_val_raw) < 5 or len(X_test_raw) < 5:
                LOGGER.warning("Skipping fold %d due to insufficient data", fold.fold_id)
                continue

            scaler = self.feature_engineer.fit_scaler(X_train_raw)
            fold.scaler = scaler
            fold.train_feature_stats = {
                "mean": X_train_raw.mean(numeric_only=True),
                "std": X_train_raw.std(ddof=0, numeric_only=True),
            }

            X_train = self.feature_engineer.scale_features(X_train_raw, scaler)
            X_val = self.feature_engineer.scale_features(X_val_raw, scaler)
            X_test = self.feature_engineer.scale_features(X_test_raw, scaler)

            params = dict(self.model_params)
            if class_weights:
                params["class_weights"] = class_weights

            model = self.model_class(params=params)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            raw_loss_history = getattr(model, "loss_history", None)
            if isinstance(raw_loss_history, dict):
                fold.loss_history = dict(raw_loss_history)
            else:
                fold.loss_history = {
                    "metric": "unknown",
                    "train": [],
                    "validation": [],
                    "best_iteration": 0,
                    "n_iterations": 0,
                    "early_stopped": False,
                }
            loss_histories.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start.isoformat(),
                    "train_end": fold.train_end.isoformat(),
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                    **fold.loss_history,
                }
            )

            fold_filter = copy.deepcopy(self.signal_filter) if self.signal_filter is not None else SignalFilter(min_confidence=self.fallback_threshold)
            val_probabilities = model.predict_proba(X_val)
            if self.threshold_optimization and len(X_val) >= self.min_threshold_optimization_samples:
                optimal_threshold = fold_filter.optimize_threshold(
                    probabilities=val_probabilities,
                    returns=returns.reindex(X_val.index).fillna(0.0),
                    thresholds=self.threshold_candidates,
                )
                fold_filter.set_threshold(optimal_threshold, "val_optimized")
            else:
                source = "fixed"
                if self.threshold_optimization and len(X_val) < self.min_threshold_optimization_samples:
                    source = "fallback_small_val"
                fold_filter.set_threshold(self.fallback_threshold, source)

            test_probabilities = model.predict_proba(X_test)
            test_signals = fold_filter.filter(test_probabilities, pd.DataFrame(index=X_test.index))
            preds = pd.Series(test_signals.astype(int), index=X_test.index, name="prediction")

            train_preds = pd.Series(model.predict(X_train), index=X_train.index, name="prediction")
            train_cls = evaluator.compute_classification_metrics(y_train, train_preds.to_numpy())
            test_cls = evaluator.compute_classification_metrics(y_test, preds.to_numpy())

            train_tm = evaluator.compute_trading_metrics(
                y_pred=train_preds,
                returns=returns.reindex(train_preds.index).fillna(0.0),
                transaction_cost_bps=self.transaction_cost_bps,
            )
            test_tm = evaluator.compute_trading_metrics(
                y_pred=preds,
                returns=returns.reindex(preds.index).fillna(0.0),
                transaction_cost_bps=self.transaction_cost_bps,
            )

            fold.train_accuracy = float(train_cls["accuracy"])
            fold.train_f1_macro = float(train_cls["f1_macro"])
            fold.train_sharpe = float(train_tm["strategy_sharpe"])
            fold.oos_accuracy = float(test_cls["accuracy"])
            fold.oos_f1_macro = float(test_cls["f1_macro"])
            fold.oos_sharpe = float(test_tm["strategy_sharpe"])
            fold.accuracy_gap = float(fold.train_accuracy - fold.oos_accuracy)
            fold.sharpe_gap = float(fold.train_sharpe - fold.oos_sharpe)
            fold.overfitting_flag = bool(fold.sharpe_gap > self.config.overfitting_threshold)
            fold.optimal_threshold = float(fold_filter.min_confidence)
            fold.threshold_source = str(getattr(fold_filter, "threshold_source", "fixed"))

            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "val_start": fold.val_start,
                    "val_end": fold.val_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "oos_accuracy": fold.oos_accuracy,
                    "oos_f1_macro": fold.oos_f1_macro,
                    "oos_sharpe": fold.oos_sharpe,
                    "accuracy_gap": fold.accuracy_gap,
                    "sharpe_gap": fold.sharpe_gap,
                    "overfitting_flag": fold.overfitting_flag,
                    "optimal_threshold": fold.optimal_threshold,
                    "threshold_source": fold.threshold_source,
                    "n_train_samples": len(X_train),
                    "n_val_samples": len(X_val),
                    "n_test_samples": len(X_test),
                    "strategy_sharpe": fold.oos_sharpe,
                }
            )
            train_fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "train_accuracy": fold.train_accuracy,
                    "train_f1_macro": fold.train_f1_macro,
                    "train_sharpe": fold.train_sharpe,
                }
            )
            thresholds.append({"fold_id": fold.fold_id, "threshold": fold.optimal_threshold, "source": fold.threshold_source})

            feature_importances.append(model.get_feature_importance())
            pred_chunks.append(preds)
            proba = test_probabilities.copy()
            proba.index = X_test.index
            proba_chunks.append(proba)

            LOGGER.info(
                "Fold %d/%d: Train %s→%s, Val %s→%s, Test %s→%s, OOS Acc %.2f, OOS Sharpe %.2f, Threshold %.2f (%s)",
                i,
                len(folds),
                fold.train_start.date(),
                fold.train_end.date(),
                fold.val_start.date(),
                fold.val_end.date(),
                fold.test_start.date(),
                fold.test_end.date(),
                fold.oos_accuracy,
                fold.oos_sharpe,
                fold.optimal_threshold,
                fold.threshold_source,
            )

        predictions = pd.concat(pred_chunks).sort_index() if pred_chunks else pd.Series(dtype=int, name="prediction")
        probabilities = pd.concat(proba_chunks).sort_index() if proba_chunks else pd.DataFrame(columns=["prob_short", "prob_flat", "prob_long"])
        metrics_per_fold = pd.DataFrame(fold_metrics)
        train_metrics_per_fold = pd.DataFrame(train_fold_metrics)

        if not predictions.empty:
            y_true_all = y.reindex(predictions.index)
            aggregate = evaluator.compute_classification_metrics(y_true_all, predictions.to_numpy())
        else:
            aggregate = {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_long": 0.0,
                "f1_short": 0.0,
                "precision_long": 0.0,
                "recall_long": 0.0,
                "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "n_samples": 0,
            }

        aggregate["model_name"] = self.model_class.__name__
        aggregate["train_window_days"] = self.config.train_window_days
        aggregate["test_window_days"] = self.config.test_window_days
        aggregate["val_fraction"] = self.config.val_fraction
        aggregate["val_purge_days"] = self.config.val_purge_days

        if feature_importances:
            fi_df = pd.concat(feature_importances, axis=1).fillna(0.0)
            fi_mean = fi_df.mean(axis=1).sort_values(ascending=False)
            fi_std = fi_df.std(axis=1).reindex(fi_mean.index)
        else:
            fi_mean = pd.Series(dtype=float)
            fi_std = pd.Series(dtype=float)

        stability = evaluator.compute_stability_risk(metrics_per_fold)
        overfit_gap = evaluator.compute_overfitting_gap(train_metrics_per_fold, metrics_per_fold)
        threshold_series = pd.Series(
            [item["threshold"] for item in thresholds],
            index=[item["fold_id"] for item in thresholds],
            dtype=float,
            name="optimal_threshold",
        ) if thresholds else pd.Series(dtype=float, name="optimal_threshold")
        threshold_stability = evaluator.compute_threshold_stability(threshold_series, source=thresholds[0]["source"] if thresholds else "fixed")
        aggregate_loss = self._compute_aggregate_loss(loss_histories)

        return WalkForwardResult(
            folds=folds,
            predictions=predictions,
            probabilities=probabilities,
            metrics_per_fold=metrics_per_fold,
            train_metrics_per_fold=train_metrics_per_fold,
            aggregate_metrics=aggregate,
            feature_importance_mean=fi_mean,
            feature_importance_std=fi_std,
            stability_risk=str(stability.get("stability_risk", "HIGH")),
            overfitting_gap=overfit_gap,
            threshold_per_fold=threshold_series,
            threshold_stability=threshold_stability,
            loss_histories=loss_histories,
            aggregate_loss=aggregate_loss,
        )

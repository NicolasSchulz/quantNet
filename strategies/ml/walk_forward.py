from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Type

import pandas as pd
from sklearn.preprocessing import RobustScaler

from strategies.ml.models.base_model import BaseModel

LOGGER = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    # window sizes expressed either in calendar days or in sample counts
    train_window_days: int = 504
    test_window_days: int = 63
    step_size_days: int = 21
    purge_days: int = 5
    embargo_days: int = 10
    # switch to interpret the above values as raw row counts instead of days
    use_days: bool = True
    def __post_init__(self):
        if not self.use_days:
            # for clarity, alias to more generic names
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
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_samples: int
    n_test_samples: int


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    predictions: pd.Series
    probabilities: pd.DataFrame
    metrics_per_fold: pd.DataFrame
    aggregate_metrics: dict[str, Any]
    feature_importance_mean: pd.Series
    feature_importance_std: pd.Series | None = None


class WalkForwardValidator:
    """Walk-forward validator with purge and embargo controls."""

    def __init__(
        self,
        model_class: Type[BaseModel],
        model_params: dict[str, Any],
        config: WalkForwardConfig,
    ) -> None:
        self.model_class = model_class
        self.model_params = model_params
        self.config = config

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

            while True:
                train_start = start_cursor
                train_end = train_start + train_td
                test_start = train_end + purge_td
                test_end = test_start + test_td

                if test_end > end_limit:
                    break

                n_train = int(((idx >= train_start) & (idx <= train_end)).sum())
                n_test = int(((idx >= test_start) & (idx <= test_end)).sum())

                if n_train > 0 and n_test > 0:
                    folds.append(
                        WalkForwardFold(
                            fold_id=fold_id,
                            train_start=train_start,
                            train_end=train_end,
                            test_start=test_start,
                            test_end=test_end,
                            n_train_samples=n_train,
                            n_test_samples=n_test,
                        )
                    )
                    fold_id += 1

                next_cursor = start_cursor + step_td
                # Enforce embargo between consecutive test windows.
                min_cursor_for_embargo = test_end + embargo_td - train_td - purge_td
                if min_cursor_for_embargo > next_cursor:
                    next_cursor = min_cursor_for_embargo

                if next_cursor >= end_limit:
                    break
                start_cursor = next_cursor
        else:
            # use sample counts rather than calendar durations
            n = len(idx)
            start_idx = 0
            while True:
                train_start_idx = start_idx
                train_end_idx = train_start_idx + self.config.train_window_days - 1
                test_start_idx = train_end_idx + self.config.purge_days + 1
                test_end_idx = test_start_idx + self.config.test_window_days - 1

                if test_end_idx >= n:
                    break

                train_start = idx[train_start_idx]
                train_end = idx[train_end_idx]
                test_start = idx[test_start_idx]
                test_end = idx[test_end_idx]

                n_train = train_end_idx - train_start_idx + 1
                n_test = test_end_idx - test_start_idx + 1

                if n_train > 0 and n_test > 0:
                    folds.append(
                        WalkForwardFold(
                            fold_id=fold_id,
                            train_start=train_start,
                            train_end=train_end,
                            test_start=test_start,
                            test_end=test_end,
                            n_train_samples=n_train,
                            n_test_samples=n_test,
                        )
                    )
                    fold_id += 1

                next_idx = start_idx + self.config.step_size_days
                # enforce embargo in index terms
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
        class_weights: dict[int, float] | None = None,
    ) -> WalkForwardResult:
        from strategies.ml.evaluator import ModelEvaluator

        data_index = X.index.intersection(y.dropna().index)
        X = X.loc[data_index].sort_index()
        y = y.loc[data_index].sort_index().astype(int)

        folds = self.generate_folds(X.index)
        evaluator = ModelEvaluator()

        pred_chunks: list[pd.Series] = []
        proba_chunks: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, Any]] = []
        feature_importances: list[pd.Series] = []

        for i, fold in enumerate(folds, start=1):
            train_mask = (X.index >= fold.train_start) & (X.index <= fold.train_end)
            test_mask = (X.index >= fold.test_start) & (X.index <= fold.test_end)

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask]
            y_test = y.loc[test_mask]

            if len(X_train) < 20 or len(X_test) < 5:
                LOGGER.warning("Skipping fold %d due to insufficient data", fold.fold_id)
                continue

            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns,
            )

            params = dict(self.model_params)
            if class_weights:
                params["class_weights"] = class_weights

            model = self.model_class(params=params)

            if len(X_test_scaled) > 21:
                X_val = X_test_scaled.tail(21)
                y_val = y_test.tail(21)
            else:
                X_val = X_test_scaled
                y_val = y_test

            model.fit(X_train_scaled, y_train, X_val=X_val, y_val=y_val)
            preds = pd.Series(model.predict(X_test_scaled), index=X_test_scaled.index, name="prediction")
            proba = model.predict_proba(X_test_scaled)

            cls_metrics = evaluator.compute_classification_metrics(y_test, preds.to_numpy())

            # Proxy fold Sharpe from directional correctness for fold stability tracking.
            proxy_ret = pd.Series((preds.values == y_test.values).astype(float) - 0.5, index=preds.index)
            fold_sharpe = float(proxy_ret.mean() / (proxy_ret.std(ddof=0) + 1e-12) * (252.0**0.5))

            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "accuracy": cls_metrics["accuracy"],
                    "f1_macro": cls_metrics["f1_macro"],
                    "f1_long": cls_metrics["f1_long"],
                    "f1_short": cls_metrics["f1_short"],
                    "fold_sharpe": fold_sharpe,
                    "n_train_samples": len(X_train),
                    "n_test_samples": len(X_test),
                }
            )

            feature_importances.append(model.get_feature_importance())
            pred_chunks.append(preds)
            proba_chunks.append(proba)

            LOGGER.info(
                "Fold %d/%d: Train %s→%s, Test %s→%s, Accuracy: %.2f, F1: %.2f",
                i,
                len(folds),
                fold.train_start.date(),
                fold.train_end.date(),
                fold.test_start.date(),
                fold.test_end.date(),
                cls_metrics["accuracy"],
                cls_metrics["f1_macro"],
            )

        if pred_chunks:
            predictions = pd.concat(pred_chunks).sort_index()
            probabilities = pd.concat(proba_chunks).sort_index()
        else:
            predictions = pd.Series(dtype=int, name="prediction")
            probabilities = pd.DataFrame(columns=["prob_short", "prob_flat", "prob_long"])

        metrics_per_fold = pd.DataFrame(fold_metrics)

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

        if feature_importances:
            fi_df = pd.concat(feature_importances, axis=1).fillna(0.0)
            fi_mean = fi_df.mean(axis=1).sort_values(ascending=False)
            fi_std = fi_df.std(axis=1).reindex(fi_mean.index)
        else:
            fi_mean = pd.Series(dtype=float)
            fi_std = pd.Series(dtype=float)

        return WalkForwardResult(
            folds=folds,
            predictions=predictions,
            probabilities=probabilities,
            metrics_per_fold=metrics_per_fold,
            aggregate_metrics=aggregate,
            feature_importance_mean=fi_mean,
            feature_importance_std=fi_std,
        )

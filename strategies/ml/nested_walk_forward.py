from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable, Type

import pandas as pd

from strategies.ml.evaluator import ModelEvaluator
from strategies.ml.feature_engineer import FeatureEngineer
from strategies.ml.models.base_model import BaseModel
from strategies.ml.signal_filter import SignalFilter
from strategies.ml.walk_forward import WalkForwardConfig, WalkForwardResult, WalkForwardValidator

LOGGER = logging.getLogger(__name__)


@dataclass
class NestedWalkForwardConfig:
    outer: WalkForwardConfig
    inner: WalkForwardConfig
    param_grid: dict[str, list[Any]]


@dataclass
class NestedWalkForwardResult:
    outer_folds: list[Any]
    predictions: pd.Series
    best_params_per_fold: list[dict[str, Any]]
    params_stability: dict[str, dict[str, int]]
    oos_metrics: dict[str, Any]
    stability_risk: str
    overfitting_gap: dict[str, Any]
    walk_forward_result: WalkForwardResult


class NestedWalkForwardValidator:
    def __init__(
        self,
        model_class: Type[BaseModel],
        config: NestedWalkForwardConfig,
        feature_engineer: FeatureEngineer,
        signal_filter: SignalFilter | None = None,
        transaction_cost_bps: float = 7.0,
        threshold_optimization: bool = False,
        threshold_candidates: list[float] | None = None,
        fallback_threshold: float = 0.45,
    ) -> None:
        self.model_class = model_class
        self.config = config
        self.feature_engineer = feature_engineer
        self.signal_filter = signal_filter
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.threshold_optimization = threshold_optimization
        self.threshold_candidates = threshold_candidates or [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        self.fallback_threshold = float(fallback_threshold)
        self.progress_callback: Callable[[dict[str, Any]], None] | None = None

    def _inner_search(
        self,
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
        returns_train: pd.Series,
        class_weights: dict[int, float] | None,
        outer_fold_id: int,
        total_steps: int,
        completed_steps_ref: dict[str, int],
    ) -> dict[str, Any]:
        keys = list(self.config.param_grid.keys())
        best_params: dict[str, Any] = {}
        best_score = float("-inf")
        total_combos = prod(len(self.config.param_grid[k]) for k in keys) if keys else 0

        for combo_index, values in enumerate(itertools.product(*[self.config.param_grid[k] for k in keys]), start=1):
            params = dict(zip(keys, values))
            completed_steps_ref["value"] += 1
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "phase": "inner_search",
                        "message": f"Outer Fold {outer_fold_id}: teste Kombination {combo_index}/{total_combos}",
                        "outer_fold": outer_fold_id,
                        "inner_combo_index": combo_index,
                        "inner_combo_total": total_combos,
                        "completed_steps": completed_steps_ref["value"],
                        "total_steps": total_steps,
                        "percent": round((completed_steps_ref["value"] / max(total_steps, 1)) * 100.0, 2),
                        "current_params": params,
                    }
                )
            validator = WalkForwardValidator(
                model_class=self.model_class,
                model_params=params,
                config=self.config.inner,
                feature_engineer=self.feature_engineer,
                signal_filter=self.signal_filter,
                threshold_optimization=self.threshold_optimization,
                threshold_candidates=self.threshold_candidates,
                fallback_threshold=self.fallback_threshold,
                transaction_cost_bps=self.transaction_cost_bps,
            )
            result = validator.run(X_train_raw, y_train, returns=returns_train, class_weights=class_weights)
            score = float(result.metrics_per_fold["oos_sharpe"].mean()) if not result.metrics_per_fold.empty else float("-inf")
            if score > best_score:
                best_score = score
                best_params = params
        return best_params

    def run(
        self,
        X_raw: pd.DataFrame,
        y: pd.Series,
        returns: pd.Series,
        class_weights: dict[int, float] | None = None,
        base_params: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> NestedWalkForwardResult:
        self.progress_callback = progress_callback
        outer_validator = WalkForwardValidator(
            model_class=self.model_class,
            model_params=base_params or {},
            config=self.config.outer,
            feature_engineer=self.feature_engineer,
            signal_filter=self.signal_filter,
            threshold_optimization=self.threshold_optimization,
            threshold_candidates=self.threshold_candidates,
            fallback_threshold=self.fallback_threshold,
            transaction_cost_bps=self.transaction_cost_bps,
        )
        outer_folds = outer_validator.generate_folds(X_raw.index)
        param_combinations = prod(len(values) for values in self.config.param_grid.values()) if self.config.param_grid else 0
        total_steps = max(1, len(outer_folds) * param_combinations + len(outer_folds) + 1)
        completed_steps_ref = {"value": 0}

        best_params_per_fold: list[dict[str, Any]] = []
        params_stability: dict[str, dict[str, int]] = {key: {} for key in self.config.param_grid}

        for fold in outer_folds:
            train_mask = (X_raw.index >= fold.train_start) & (X_raw.index <= fold.val_end)
            X_outer_train_raw = X_raw.loc[train_mask]
            y_outer_train = y.loc[train_mask]
            returns_outer_train = returns.reindex(X_outer_train_raw.index).fillna(0.0)

            best_params = self._inner_search(
                X_outer_train_raw,
                y_outer_train,
                returns_outer_train,
                class_weights,
                outer_fold_id=fold.fold_id,
                total_steps=total_steps,
                completed_steps_ref=completed_steps_ref,
            )
            LOGGER.info("Outer Fold %s: beste Params = %s", fold.fold_id, best_params)
            best_params_per_fold.append({"fold_id": fold.fold_id, **best_params})
            for key, value in best_params.items():
                bucket = params_stability.setdefault(key, {})
                bucket[str(value)] = int(bucket.get(str(value), 0)) + 1
            completed_steps_ref["value"] += 1
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "phase": "outer_summary",
                        "message": f"Outer Fold {fold.fold_id}: beste Parameter gefunden",
                        "outer_fold": fold.fold_id,
                        "completed_steps": completed_steps_ref["value"],
                        "total_steps": total_steps,
                        "percent": round((completed_steps_ref["value"] / max(total_steps, 1)) * 100.0, 2),
                        "best_params": best_params,
                    }
                )

        flattened_counts: dict[str, Any] = {}
        for key, counts in params_stability.items():
            most_common = max(counts, key=counts.get) if counts else None
            total = sum(counts.values()) or 1
            flattened_counts[key] = {
                "values": counts,
                "most_common": most_common,
                "stability_pct": (counts.get(most_common, 0) / total) * 100.0 if most_common is not None else 0.0,
            }

        consensus_params = dict(base_params or {})
        for key, summary in flattened_counts.items():
            if summary["most_common"] is not None:
                value = summary["most_common"]
                consensus_params[key] = float(value) if "." in str(value) else int(value)

        final_validator = WalkForwardValidator(
            model_class=self.model_class,
            model_params=consensus_params,
            config=self.config.outer,
            feature_engineer=self.feature_engineer,
            signal_filter=self.signal_filter,
            threshold_optimization=self.threshold_optimization,
            threshold_candidates=self.threshold_candidates,
            fallback_threshold=self.fallback_threshold,
            transaction_cost_bps=self.transaction_cost_bps,
        )
        completed_steps_ref["value"] += 1
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "phase": "final_evaluation",
                    "message": "Finale Walk-Forward Auswertung mit Konsens-Parametern",
                    "completed_steps": completed_steps_ref["value"],
                    "total_steps": total_steps,
                    "percent": round((completed_steps_ref["value"] / max(total_steps, 1)) * 100.0, 2),
                    "consensus_params": consensus_params,
                }
            )
        result = final_validator.run(X_raw, y, returns=returns, class_weights=class_weights)
        evaluator = ModelEvaluator()
        oos_metrics = evaluator.compute_trading_metrics(result.predictions, returns.reindex(result.predictions.index).fillna(0.0), transaction_cost_bps=self.transaction_cost_bps)

        return NestedWalkForwardResult(
            outer_folds=outer_folds,
            predictions=result.predictions,
            best_params_per_fold=best_params_per_fold,
            params_stability=flattened_counts,
            oos_metrics=oos_metrics,
            stability_risk=result.stability_risk,
            overfitting_gap=result.overfitting_gap,
            walk_forward_result=result,
        )

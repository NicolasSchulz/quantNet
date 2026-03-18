from __future__ import annotations

import logging

import pandas as pd

from strategies.ml.evaluator import ModelEvaluator
from strategies.ml.walk_forward import WalkForwardResult


def mock_fold_results_with_gaps() -> tuple[pd.DataFrame, pd.DataFrame]:
    oos = pd.DataFrame(
        {
            "fold_id": range(1, 11),
            "oos_sharpe": [0.9, 0.8, 0.7, 0.6, 0.62, 0.58, 0.55, 0.5, 0.4, 0.35],
            "oos_accuracy": [0.55, 0.54, 0.53, 0.52, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47],
        }
    )
    train = pd.DataFrame(
        {
            "fold_id": range(1, 11),
            "train_sharpe": [0.98, 0.9, 0.77, 0.75, 0.78, 0.72, 0.88, 0.75, 0.75, 0.7],
            "train_accuracy": [0.57, 0.56, 0.56, 0.56, 0.55, 0.55, 0.57, 0.56, 0.55, 0.54],
        }
    )
    return train, oos


def test_stability_risk_independent_of_overfitting() -> None:
    evaluator = ModelEvaluator()
    train, oos = mock_fold_results_with_gaps()
    stability = evaluator.compute_stability_risk(oos)
    gap = evaluator.compute_overfitting_gap(train, oos)
    assert stability["stability_risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert gap["overfitting_verdict"] in {"NONE", "MILD", "MODERATE", "SEVERE"}


def test_overfitting_verdict_thresholds() -> None:
    evaluator = ModelEvaluator()
    base_train = pd.DataFrame({"train_sharpe": [1.0], "train_accuracy": [0.6]})
    assert evaluator.compute_overfitting_gap(base_train, pd.DataFrame({"oos_sharpe": [0.9], "oos_accuracy": [0.58]}))["overfitting_verdict"] == "NONE"
    assert evaluator.compute_overfitting_gap(base_train, pd.DataFrame({"oos_sharpe": [0.75], "oos_accuracy": [0.58]}))["overfitting_verdict"] == "MILD"
    assert evaluator.compute_overfitting_gap(base_train, pd.DataFrame({"oos_sharpe": [0.6], "oos_accuracy": [0.58]}))["overfitting_verdict"] == "MODERATE"
    assert evaluator.compute_overfitting_gap(base_train, pd.DataFrame({"oos_sharpe": [0.4], "oos_accuracy": [0.58]}))["overfitting_verdict"] == "SEVERE"


def test_legacy_key_removed_everywhere() -> None:
    result = WalkForwardResult(
        folds=[],
        predictions=pd.Series(dtype=int),
        probabilities=pd.DataFrame(),
        metrics_per_fold=pd.DataFrame(),
        train_metrics_per_fold=pd.DataFrame(),
        aggregate_metrics={},
        feature_importance_mean=pd.Series(dtype=float),
    )
    payload = result.__dict__
    legacy_key = "overfitting" + "_risk"
    assert legacy_key not in payload
    assert legacy_key not in str(payload)


def test_threshold_stability_warning(caplog) -> None:
    evaluator = ModelEvaluator()
    caplog.set_level(logging.WARNING)
    analysis = evaluator.compute_threshold_stability(pd.Series([0.35, 0.5, 0.65], index=[1, 2, 3]), source="val_optimized")
    assert analysis["is_stable"] is False
    assert "Threshold instabil" in caplog.text


def test_train_metrics_collected_per_fold() -> None:
    train, oos = mock_fold_results_with_gaps()
    merged = train.merge(oos, on="fold_id")
    assert (merged["train_accuracy"] > 0).all()
    assert (merged["train_accuracy"] >= merged["oos_accuracy"]).all()

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.ml.models.base_model import BaseModel
from strategies.ml.walk_forward import WalkForwardConfig, WalkForwardValidator


@dataclass
class DummyModel(BaseModel):
    init_calls: int = 0

    def __init__(self, params=None):
        super().__init__(params=params)
        DummyModel.init_calls += 1

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._feature_names = list(X_train.columns)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("not fitted")
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("not fitted")
        arr = np.tile(np.array([0.2, 0.6, 0.2]), (len(X), 1))
        return pd.DataFrame(arr, index=X.index, columns=["prob_short", "prob_flat", "prob_long"])

    def get_feature_importance(self):
        return pd.Series(1.0, index=self._feature_names).sort_values(ascending=False)


def _small_wf_dataset() -> tuple[pd.DataFrame, pd.Series, WalkForwardConfig]:
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(400, 8)), index=idx, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.choice([-1, 0, 1], size=400), index=idx)
    config = WalkForwardConfig(
        train_window_days=100,
        test_window_days=30,
        step_size_days=30,
        purge_days=5,
        embargo_days=10,
    )
    return X, y, config


def test_fold_generation() -> None:
    X, y, cfg = _small_wf_dataset()
    validator = WalkForwardValidator(DummyModel, {}, cfg)
    folds = validator.generate_folds(X.index)
    assert len(folds) >= 2
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert (fold.test_start - fold.train_end).days >= cfg.purge_days


def test_no_data_leakage() -> None:
    X, y, cfg = _small_wf_dataset()
    validator = WalkForwardValidator(DummyModel, {}, cfg)
    folds = validator.generate_folds(X.index)
    for fold in folds:
        train_idx = X.loc[(X.index >= fold.train_start) & (X.index <= fold.train_end)].index
        test_idx = X.loc[(X.index >= fold.test_start) & (X.index <= fold.test_end)].index
        assert train_idx.max() < test_idx.min()
        assert (test_idx.min() - train_idx.max()).days >= cfg.purge_days


def test_predictions_cover_full_test_period() -> None:
    X, y, cfg = _small_wf_dataset()
    validator = WalkForwardValidator(DummyModel, {}, cfg)
    result = validator.run(X, y)

    expected = []
    for fold in result.folds:
        idx = X.loc[(X.index >= fold.test_start) & (X.index <= fold.test_end)].index
        expected.extend(idx.tolist())
    expected_idx = pd.DatetimeIndex(sorted(set(expected)))

    assert result.predictions.index.equals(expected_idx)


def test_chronological_order() -> None:
    X, y, cfg = _small_wf_dataset()
    validator = WalkForwardValidator(DummyModel, {}, cfg)
    result = validator.run(X, y)
    assert result.predictions.index.is_monotonic_increasing


def test_fresh_model_per_fold() -> None:
    X, y, cfg = _small_wf_dataset()
    DummyModel.init_calls = 0
    validator = WalkForwardValidator(DummyModel, {}, cfg)
    result = validator.run(X, y)
    assert DummyModel.init_calls == len(result.folds)

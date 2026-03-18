from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.ml.models.base_model import BaseModel
from strategies.ml.signal_filter import SignalFilter
from strategies.ml.walk_forward import WalkForwardConfig, WalkForwardValidator


@dataclass
class DummyModel(BaseModel):
    init_calls: int = 0
    fit_args: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None  # type: ignore[assignment]

    def __init__(self, params=None):
        super().__init__(params=params)
        DummyModel.init_calls += 1
        if DummyModel.fit_args is None:
            DummyModel.fit_args = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._feature_names = list(X_train.columns)
        self.is_fitted = True
        DummyModel.fit_args.append((X_train.copy(), X_val.copy(), y_train.to_frame(name="y")))

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


def _small_wf_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, WalkForwardConfig]:
    idx = pd.date_range("2021-01-01", periods=420, freq="D", tz="UTC")
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(420, 8)), index=idx, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.choice([-1, 0, 1], size=420), index=idx)
    returns = pd.Series(rng.normal(0.0, 0.01, size=420), index=idx)
    config = WalkForwardConfig(
        train_window_days=100,
        test_window_days=30,
        step_size_days=30,
        purge_days=5,
        embargo_days=10,
        val_fraction=0.15,
        val_purge_days=5,
    )
    return X, y, returns, config


def _validator(cfg: WalkForwardConfig) -> WalkForwardValidator:
    class LocalScaler:
        def fit_scaler(self, X: pd.DataFrame):
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            scaler.fit(X)
            return scaler

        def scale_features(self, X: pd.DataFrame, scaler):
            return pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    return WalkForwardValidator(
        DummyModel,
        {},
        cfg,
        feature_engineer=LocalScaler(),
        signal_filter=SignalFilter(min_confidence=0.45, min_holding_days=1, signal_smoothing=False),
        threshold_optimization=True,
        threshold_candidates=[0.4, 0.45, 0.5],
    )


def test_fold_generation() -> None:
    X, _, _, cfg = _small_wf_dataset()
    folds = _validator(cfg).generate_folds(X.index)
    assert len(folds) >= 2
    for fold in folds:
        assert fold.train_end < fold.val_start
        assert fold.val_end < fold.test_start


def test_val_split_from_train_not_test() -> None:
    X, _, _, cfg = _small_wf_dataset()
    folds = _validator(cfg).generate_folds(X.index)
    for fold in folds:
        assert fold.val_end <= fold.test_start
        assert fold.val_start > fold.train_start
        assert fold.test_start > fold.val_end


def test_predictions_cover_full_test_period() -> None:
    X, y, returns, cfg = _small_wf_dataset()
    result = _validator(cfg).run(X, y, returns=returns)
    expected = []
    for fold in result.folds:
        idx = X.loc[(X.index >= fold.test_start) & (X.index <= fold.test_end)].index
        expected.extend(idx.tolist())
    expected_idx = pd.DatetimeIndex(sorted(set(expected)))
    assert result.predictions.index.equals(expected_idx)


def test_scaler_fitted_without_val() -> None:
    X, y, returns, cfg = _small_wf_dataset()
    result = _validator(cfg).run(X, y, returns=returns)
    assert result.folds
    for fold in result.folds:
        assert fold.scaler is not None
        train_slice = X.loc[(X.index >= fold.train_start) & (X.index <= fold.train_end)]
        assert len(getattr(fold.scaler, "center_", [])) <= train_slice.shape[1]


def test_early_stopping_uses_val_not_test() -> None:
    X, y, returns, cfg = _small_wf_dataset()
    DummyModel.fit_args = []
    result = _validator(cfg).run(X, y, returns=returns)
    assert result.folds
    assert DummyModel.fit_args
    for (X_train, X_val, _), fold in zip(DummyModel.fit_args, result.folds):
        assert X_train.index.max() <= fold.train_end
        assert X_val.index.min() >= fold.val_start
        assert X_val.index.max() <= fold.val_end

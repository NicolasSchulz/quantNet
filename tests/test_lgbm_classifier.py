from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

pytest.importorskip("lightgbm")

from strategies.ml.models.lgbm_classifier import FeatureMismatchError, LGBMClassifier


@pytest.fixture
def synthetic_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y_raw = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    mapping = {0: -1, 1: 0, 2: 1}
    y = pd.Series([mapping[int(v)] for v in y_raw], name="label")
    X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(20)], dtype="float64")
    return X_df, y


def test_fit_predict_shape(synthetic_classification_data) -> None:
    X, y = synthetic_classification_data
    model = LGBMClassifier()
    model.fit(X.iloc[:400], y.iloc[:400], X_val=X.iloc[400:], y_val=y.iloc[400:])
    pred = model.predict(X.iloc[400:])
    assert pred.shape == (len(X.iloc[400:]),)


def test_predict_proba_columns(synthetic_classification_data) -> None:
    X, y = synthetic_classification_data
    model = LGBMClassifier()
    model.fit(X.iloc[:400], y.iloc[:400])
    proba = model.predict_proba(X.iloc[400:])
    assert list(proba.columns) == ["prob_short", "prob_flat", "prob_long"]
    assert ((proba >= 0.0) & (proba <= 1.0)).all().all()
    assert np.allclose(proba.sum(axis=1).values, 1.0, atol=1e-6)


def test_not_fitted_error(synthetic_classification_data) -> None:
    X, _ = synthetic_classification_data
    model = LGBMClassifier()
    with pytest.raises(NotFittedError):
        model.predict(X.iloc[:5])


def test_feature_mismatch_error(synthetic_classification_data) -> None:
    X, y = synthetic_classification_data
    model = LGBMClassifier()
    model.fit(X.iloc[:400], y.iloc[:400])
    bad = X.iloc[400:].copy()
    bad.columns = [f"x_{i}" for i in range(len(bad.columns))]
    with pytest.raises(FeatureMismatchError):
        model.predict_proba(bad)


def test_feature_importance_sorted(synthetic_classification_data) -> None:
    X, y = synthetic_classification_data
    model = LGBMClassifier()
    model.fit(X.iloc[:400], y.iloc[:400])
    fi = model.get_feature_importance()
    assert len(fi) == X.shape[1]
    assert fi.is_monotonic_decreasing


def test_save_load_roundtrip(synthetic_classification_data, tmp_path) -> None:
    X, y = synthetic_classification_data
    model = LGBMClassifier()
    model.fit(X.iloc[:400], y.iloc[:400])
    pred1 = model.predict(X.iloc[400:])

    p = tmp_path / "lgbm.joblib"
    model.save(str(p))

    loaded = LGBMClassifier()
    loaded.load(str(p))
    pred2 = loaded.predict(X.iloc[400:])
    assert np.array_equal(pred1, pred2)


def test_label_encoding(synthetic_classification_data) -> None:
    X, y = synthetic_classification_data
    model = LGBMClassifier()
    model.fit(X.iloc[:400], y.iloc[:400])
    pred = model.predict(X.iloc[400:])
    assert set(np.unique(pred)).issubset({-1, 0, 1})

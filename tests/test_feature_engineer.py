from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas_ta")

from strategies.ml.feature_engineer import FeatureEngineer


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 320
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, n))
    volume = rng.integers(500_000, 2_000_000, n)

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _engineer() -> FeatureEngineer:
    return FeatureEngineer(config={"feature_groups": ["trend", "momentum", "volatility", "volume", "candle"], "normalization": "robust", "warmup_bars": 200})


def test_compute_features_no_scaling(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = _engineer()
    out = eng.compute_features(synthetic_ohlcv)
    assert len(out) < len(synthetic_ohlcv)
    assert not out.isna().any().any()
    assert float(out.mean(numeric_only=True).abs().mean()) > 0.01
    assert out["rsi_14"].between(0.0, 100.0).all()


def test_fit_scaler_only_on_train(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = _engineer()
    features = eng.compute_features(synthetic_ohlcv)
    X_train = features.iloc[:50]
    X_test = features.iloc[50:80]
    scaler = eng.fit_scaler(X_train)
    scaled_test = eng.scale_features(X_test, scaler)
    assert len(getattr(scaler, "center_", [])) == len([c for c in X_train.columns if c not in {"rsi_14", "rsi_28", "stoch_k", "stoch_d", "bb_position", "body_size", "upper_wick", "lower_wick"}])
    assert scaled_test.shape == X_test.shape


def test_scale_features_deterministic(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = _engineer()
    features = eng.compute_features(synthetic_ohlcv)
    scaler = eng.fit_scaler(features.iloc[:60])
    once = eng.scale_features(features.iloc[60:90], scaler)
    twice = eng.scale_features(features.iloc[60:90], scaler)
    pd.testing.assert_frame_equal(once, twice)


def test_no_leakage_across_folds(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = _engineer()
    features = eng.compute_features(synthetic_ohlcv)
    scaler_1 = eng.fit_scaler(features.iloc[:50])
    scaler_2 = eng.fit_scaler(features.iloc[50:100])
    assert not np.allclose(np.asarray(scaler_1.center_), np.asarray(scaler_2.center_))


def test_structure_features_are_numeric(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = FeatureEngineer(
        config={
            "feature_groups": ["trend", "momentum", "volatility", "volume", "candle", "structure"],
            "normalization": "robust",
            "warmup_bars": 200,
            "mss_strategy": {"swing_n": 2, "atr_period": 14, "tp_mult": 3.0, "sl_mult": 1.5, "max_hold": 24, "pullback_timeout": 4},
        }
    )
    features = eng.compute_features(synthetic_ohlcv)
    for col in ["mss_candidate", "mss_direction", "market_structure_bias", "choch_bullish_flag", "choch_bearish_flag"]:
        assert col in features.columns
        assert features[col].dtype == np.float64

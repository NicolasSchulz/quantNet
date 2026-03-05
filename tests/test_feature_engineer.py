from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas_ta")

from strategies.ml.feature_engineer import BOUNDED_FEATURES, FeatureEngineer


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, n))
    volume = rng.integers(500_000, 2_000_000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_output_shape(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = FeatureEngineer(config={"feature_groups": ["trend", "momentum", "volatility", "volume", "candle"], "normalization": "robust", "warmup_bars": 200})
    out = eng.transform(synthetic_ohlcv)
    assert len(out) < len(synthetic_ohlcv)
    assert not out.isna().any().any()


def test_feature_names_consistent(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = FeatureEngineer(config={"feature_groups": ["trend", "momentum", "volatility", "volume", "candle"], "normalization": "robust", "warmup_bars": 200})
    out = eng.transform(synthetic_ohlcv)
    assert eng.get_feature_names() == list(out.columns)


def test_no_lookahead(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = FeatureEngineer(config={"feature_groups": ["trend", "momentum", "volatility", "volume", "candle"], "normalization": "none", "warmup_bars": 200})
    subset = synthetic_ohlcv.iloc[:260]
    out_subset = eng.transform(subset)
    out_full = eng.transform(synthetic_ohlcv)
    ts = out_subset.index[-1]
    pd.testing.assert_series_equal(out_subset.loc[ts], out_full.loc[ts], check_names=False)


def test_normalization_range(synthetic_ohlcv: pd.DataFrame) -> None:
    eng = FeatureEngineer(config={"feature_groups": ["trend", "momentum", "volatility", "volume", "candle"], "normalization": "robust", "warmup_bars": 200})
    out = eng.fit_transform(synthetic_ohlcv)

    non_bounded = [c for c in out.columns if c not in BOUNDED_FEATURES]
    if non_bounded:
        means = out[non_bounded].mean().abs()
        assert float(means.mean()) < 1.0

    for col in ["rsi_14", "rsi_28", "stoch_k", "stoch_d", "bb_position"]:
        assert col in out.columns
        assert out[col].between(-1e-6, 100.0 + 1e-6).all()


def test_scaler_persistence(synthetic_ohlcv: pd.DataFrame, tmp_path) -> None:
    eng = FeatureEngineer(config={"feature_groups": ["trend", "momentum", "volatility", "volume", "candle"], "normalization": "robust", "warmup_bars": 200})
    fitted = eng.fit_transform(synthetic_ohlcv)

    scaler_path = tmp_path / "robust_scaler.joblib"
    eng.save_scaler(str(scaler_path))

    eng2 = FeatureEngineer(config={"feature_groups": ["trend", "momentum", "volatility", "volume", "candle"], "normalization": "robust", "warmup_bars": 200})
    eng2.load_scaler(str(scaler_path))
    transformed = eng2.transform(synthetic_ohlcv)

    pd.testing.assert_frame_equal(fitted, transformed)

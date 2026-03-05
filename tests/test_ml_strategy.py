from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from strategies.ml.ml_strategy import MLStrategy
from strategies.ml.signal_filter import SignalFilter


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, len(idx)))
    open_ = close * (1 + rng.normal(0, 0.001, len(idx)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.01, len(idx)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.01, len(idx)))
    vol = rng.integers(500_000, 2_000_000, len(idx))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


@pytest.fixture
def mock_model(synthetic_ohlcv: pd.DataFrame) -> MagicMock:
    model = MagicMock()
    probs = pd.DataFrame(
        {
            "prob_short": 0.1,
            "prob_flat": 0.3,
            "prob_long": 0.6,
        },
        index=synthetic_ohlcv.index,
    )
    model.predict_proba.return_value = probs
    return model


@pytest.fixture
def mock_registry(mock_model: MagicMock) -> MagicMock:
    registry = MagicMock()
    registry.get_default_model.return_value = (mock_model, "v1")
    registry._read.return_value = {
        "models": {
            "mock_default": {
                "metrics": {"sharpe": 1.0},
                "model_path": "./models/mock.joblib",
                "feature_version": "v1",
            }
        },
        "default": {"SPY": "mock_default"},
    }
    return registry


@pytest.fixture
def mock_feature_engineer(synthetic_ohlcv: pd.DataFrame) -> MagicMock:
    fe = MagicMock()
    feats = pd.DataFrame({f"f{i}": np.linspace(0.0, 1.0, len(synthetic_ohlcv)) for i in range(5)}, index=synthetic_ohlcv.index)
    fe.transform.return_value = feats
    fe.warmup_bars = 200
    fe.scaler = object()
    return fe


def test_generate_signals_shape(synthetic_ohlcv, mock_registry, mock_feature_engineer) -> None:
    strategy = MLStrategy(
        symbol="SPY",
        model_registry=mock_registry,
        feature_engineer=mock_feature_engineer,
        signal_filter=SignalFilter(min_confidence=0.45),
    )
    signals = strategy.generate_signals(synthetic_ohlcv)
    assert signals.index.equals(synthetic_ohlcv.index)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_no_lookahead_in_signals(synthetic_ohlcv, mock_registry, mock_feature_engineer) -> None:
    strategy = MLStrategy(
        symbol="SPY",
        model_registry=mock_registry,
        feature_engineer=mock_feature_engineer,
        signal_filter=SignalFilter(min_confidence=0.45),
    )
    _ = strategy.generate_signals(synthetic_ohlcv)
    mock_feature_engineer.transform.assert_called_once()
    called_df = mock_feature_engineer.transform.call_args[0][0]
    assert called_df.index.equals(synthetic_ohlcv.index)


def test_signal_uses_transform_not_fit_transform(synthetic_ohlcv, mock_registry, mock_feature_engineer) -> None:
    strategy = MLStrategy(
        symbol="SPY",
        model_registry=mock_registry,
        feature_engineer=mock_feature_engineer,
        signal_filter=SignalFilter(min_confidence=0.45),
    )
    _ = strategy.generate_signals(synthetic_ohlcv)
    mock_feature_engineer.transform.assert_called_once()
    assert not mock_feature_engineer.fit_transform.called


def test_get_parameters_complete(mock_registry, mock_feature_engineer) -> None:
    strategy = MLStrategy(
        symbol="SPY",
        model_registry=mock_registry,
        feature_engineer=mock_feature_engineer,
        signal_filter=SignalFilter(min_confidence=0.45),
    )
    params = strategy.get_parameters()
    expected = {"model_id", "symbol", "feature_version", "min_confidence", "regime_filter", "min_holding_days", "model_metrics"}
    assert expected.issubset(set(params.keys()))


def test_warmup_bars(mock_registry, mock_feature_engineer) -> None:
    strategy = MLStrategy(
        symbol="SPY",
        model_registry=mock_registry,
        feature_engineer=mock_feature_engineer,
        signal_filter=SignalFilter(min_confidence=0.45),
    )
    assert strategy.warmup_bars() == 200


def test_fit_logs_warning(caplog, synthetic_ohlcv, mock_registry, mock_feature_engineer) -> None:
    strategy = MLStrategy(
        symbol="SPY",
        model_registry=mock_registry,
        feature_engineer=mock_feature_engineer,
        signal_filter=SignalFilter(min_confidence=0.45),
    )
    with caplog.at_level("WARNING"):
        strategy.fit(synthetic_ohlcv)
    assert any("hat keinen Effekt" in rec.message for rec in caplog.records)


def test_generate_signal_single_metadata(synthetic_ohlcv, mock_registry, mock_feature_engineer) -> None:
    strategy = MLStrategy(
        symbol="SPY",
        model_registry=mock_registry,
        feature_engineer=mock_feature_engineer,
        signal_filter=SignalFilter(min_confidence=0.45),
    )
    signal, meta = strategy.generate_signal_single(synthetic_ohlcv)
    assert signal in {-1, 0, 1}
    keys = {"prob_long", "prob_short", "prob_flat", "confidence", "regime", "filtered_by"}
    assert keys.issubset(set(meta.keys()))

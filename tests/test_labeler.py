from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategies.ml.labeler import TripleBarrierLabeler


def _base_ohlcv(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = np.full(n, 100.0)
    open_ = close.copy()
    high = close * 1.001
    low = close * 0.999
    volume = np.full(n, 1_000_000)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_tp_label() -> None:
    df = _base_ohlcv()
    df.iloc[1, df.columns.get_loc("high")] = 105.0
    labeler = TripleBarrierLabeler(take_profit=0.02, stop_loss=0.01, max_holding=10)
    labels = labeler.label(df)
    assert labels.iloc[0] == 1.0


def test_sl_label() -> None:
    df = _base_ohlcv()
    df.iloc[1, df.columns.get_loc("low")] = 95.0
    labeler = TripleBarrierLabeler(take_profit=0.02, stop_loss=0.01, max_holding=10)
    labels = labeler.label(df)
    assert labels.iloc[0] == -1.0


def test_time_barrier() -> None:
    df = _base_ohlcv()
    labeler = TripleBarrierLabeler(take_profit=0.05, stop_loss=0.05, max_holding=5)
    labels = labeler.label(df)
    assert labels.iloc[0] == 0.0


def test_nan_at_end() -> None:
    df = _base_ohlcv(n=25)
    labeler = TripleBarrierLabeler(max_holding=7)
    labels = labeler.label(df)
    assert labels.iloc[-7:].isna().all()


def test_label_distribution_warning(caplog) -> None:
    df = _base_ohlcv(n=40)
    df.iloc[1:20, df.columns.get_loc("high")] = 110.0
    labeler = TripleBarrierLabeler(take_profit=0.01, stop_loss=0.2, max_holding=3)
    labels = labeler.label(df)

    with caplog.at_level(logging.WARNING):
        _ = labeler.label_distribution(labels)
    assert any("Class imbalance detected" in rec.message for rec in caplog.records)


def test_no_future_leakage() -> None:
    df = _base_ohlcv(n=30)
    labeler = TripleBarrierLabeler(max_holding=5)

    labels_full = labeler.label(df)
    df_truncated = df.iloc[:20]
    labels_truncated = labeler.label(df_truncated)

    # For timestamps with full forward window in both datasets, labels must match.
    shared_idx = labels_truncated.index[:15]
    pd.testing.assert_series_equal(labels_truncated.loc[shared_idx], labels_full.loc[shared_idx], check_names=False)

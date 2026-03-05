from __future__ import annotations

import pandas as pd
import pytest

from data.normalizer import normalize_ohlcv


@pytest.fixture
def raw_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )


def test_normalize_schema_and_utc(raw_df: pd.DataFrame) -> None:
    out = normalize_ohlcv(raw_df, symbol="spy", asset_class="etf")
    assert list(out.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "symbol",
        "asset_class",
    ]
    assert str(out.index.tz) == "UTC"
    assert (out["symbol"] == "SPY").all()


def test_normalize_raises_on_nan_ohlc(raw_df: pd.DataFrame) -> None:
    raw_df.loc[raw_df.index[1], "Close"] = None
    with pytest.raises(ValueError, match="OHLC"):
        normalize_ohlcv(raw_df, symbol="SPY", asset_class="etf")


def test_normalize_raises_on_negative_volume(raw_df: pd.DataFrame) -> None:
    raw_df.loc[raw_df.index[0], "Volume"] = -1
    with pytest.raises(ValueError, match="Volume"):
        normalize_ohlcv(raw_df, symbol="SPY", asset_class="etf")


def test_normalize_polygon_epoch_milliseconds() -> None:
    ts_ms = [1704067200000, 1704070800000]
    raw = pd.DataFrame(
        {
            "timestamp": ts_ms,
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1200],
            "vwap": [100.2, 101.2],
        }
    )

    out = normalize_ohlcv(raw, symbol="SPY", asset_class="etf")

    assert out.index[0] == pd.Timestamp("2024-01-01 00:00:00+00:00")
    assert out.index[1] == pd.Timestamp("2024-01-01 01:00:00+00:00")

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("binance")

from data.ingestion.binance_feed import BinanceFeed, NoDataError
from data.ingestion.feed_factory import FeedFactory
from data.normalizer import normalize_ohlcv


class FakeClient:
    KLINE_INTERVAL_1HOUR = "1h"
    KLINE_INTERVAL_1DAY = "1d"
    KLINE_INTERVAL_15MINUTE = "15m"
    KLINE_INTERVAL_4HOUR = "4h"

    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs
        self.get_historical_klines = lambda **kwargs: []
        self.get_klines = lambda **kwargs: []
        self.ping = lambda: {}


@pytest.fixture
def mock_binance_klines() -> list[list[object]]:
    start = pd.Timestamp("2024-01-05T00:00:00Z")
    out: list[list[object]] = []
    for i in range(100):
        ts = int((start + pd.Timedelta(hours=i)).timestamp() * 1000)
        close = 40_000 + i * 50
        out.append(
            [
                ts,
                str(close - 25),
                str(close + 50),
                str(close - 75),
                str(close),
                str(15 + i / 10),
                ts + 3_599_999,
                "0",
                100 + i,
                str(7 + i / 20),
                "0",
                "0",
            ]
        )
    return out


def test_fetch_historical_output_schema(monkeypatch, mock_binance_klines: list[list[object]]) -> None:
    fake_client = SimpleNamespace(
        get_historical_klines=lambda **kwargs: mock_binance_klines,
        ping=lambda: {},
    )
    monkeypatch.setattr("data.ingestion.binance_feed.Client", FakeClient)
    monkeypatch.setattr("data.ingestion.binance_feed.time.sleep", lambda *_args, **_kwargs: None)

    feed = BinanceFeed()
    feed.client = fake_client
    df = feed.fetch_historical("BTCUSDT", "2024-01-01", "2024-01-10", "1h")

    assert {"open", "high", "low", "close", "volume", "number_of_trades", "taker_buy_base_volume"}.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_interval_mapping() -> None:
    assert BinanceFeed._interval_to_binance("1h") == BinanceFeed._interval_to_binance("1h")
    assert BinanceFeed._interval_to_binance("1d") == BinanceFeed._interval_to_binance("1d")
    with pytest.raises(ValueError):
        BinanceFeed._interval_to_binance("xyz")


def test_pagination_multiple_batches(monkeypatch, mock_binance_klines: list[list[object]]) -> None:
    first_batch = mock_binance_klines * 10
    first_batch = first_batch[:1000]
    second_batch = mock_binance_klines[:25]
    calls: list[object] = []

    def get_historical_klines(**kwargs):
        calls.append(kwargs["start_str"])
        return first_batch if len(calls) == 1 else second_batch

    fake_client = SimpleNamespace(get_historical_klines=get_historical_klines, ping=lambda: {})
    monkeypatch.setattr("data.ingestion.binance_feed.Client", FakeClient)
    monkeypatch.setattr("data.ingestion.binance_feed.time.sleep", lambda *_args, **_kwargs: None)

    feed = BinanceFeed()
    feed.client = fake_client
    df = feed.fetch_historical("BTCUSDT", "2024-01-01", "2024-01-10", "1h")
    assert len(calls) == 2
    assert len(df) == 1025


def test_24_7_no_gaps(monkeypatch) -> None:
    weekend_start = pd.Timestamp("2024-01-06T00:00:00Z")
    klines = []
    for i in range(48):
        ts = int((weekend_start + pd.Timedelta(hours=i)).timestamp() * 1000)
        klines.append([ts, "1", "2", "0.5", "1.5", "100", ts + 1, "0", 10, "50", "0", "0"])

    fake_client = SimpleNamespace(get_historical_klines=lambda **kwargs: klines, ping=lambda: {})
    monkeypatch.setattr("data.ingestion.binance_feed.Client", FakeClient)
    monkeypatch.setattr("data.ingestion.binance_feed.time.sleep", lambda *_args, **_kwargs: None)

    feed = BinanceFeed()
    feed.client = fake_client
    df = feed.fetch_historical("BTCUSDT", "2024-01-06", "2024-01-08", "1h")
    assert df.index.dayofweek.isin([5, 6]).any()
    assert len(df) == 48


def test_crypto_symbol_detection_in_factory(monkeypatch) -> None:
    monkeypatch.setattr("data.ingestion.binance_feed.Client", FakeClient)
    crypto_feed = FeedFactory.create_for_symbol("BTCUSDT", config={"data": {"feed": {"primary": "yahoo"}, "binance": {}}})
    equity_feed = FeedFactory.create_for_symbol("SPY", config={"data": {"feed": {"primary": "yahoo"}}})
    assert crypto_feed.__class__.__name__ == "BinanceFeed"
    assert equity_feed.__class__.__name__ == "YahooFeed"


def test_normalizer_sets_crypto_asset_class(mock_binance_klines: list[list[object]]) -> None:
    raw = BinanceFeed._bars_to_frame(mock_binance_klines)
    normalized = normalize_ohlcv(raw, "BTCUSDT", "equity", interval="1h")
    assert (normalized["asset_class"] == "crypto").all()


def test_empty_response_raises(monkeypatch) -> None:
    fake_client = SimpleNamespace(get_historical_klines=lambda **kwargs: [], ping=lambda: {})
    monkeypatch.setattr("data.ingestion.binance_feed.Client", FakeClient)

    feed = BinanceFeed()
    feed.client = fake_client
    with pytest.raises(NoDataError):
        feed.fetch_historical("BTCUSDT", "2024-01-01", "2024-01-10", "1h")

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("polygon")

from data.ingestion.feed_factory import FeedFactory, MissingApiKeyError
from data.ingestion.polygon_feed import NoDataError, PolygonFeed, PolygonRateLimitError, SymbolNotFoundError
from data.ingestion.yahoo_feed import YahooFeed


class FakeBar:
    def __init__(self, o, h, l, c, v, vw, t, n):
        self.o = o
        self.h = h
        self.l = l
        self.c = c
        self.v = v
        self.vw = vw
        self.t = t
        self.n = n


def _bars(n=20):
    start = pd.Timestamp("2024-01-01T09:30:00Z")
    out = []
    for i in range(n):
        ts = int((start + pd.Timedelta(hours=i)).timestamp() * 1000)
        out.append(FakeBar(100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i, 100.2 + i, ts, 10 + i))
    return out


def test_interval_mapping() -> None:
    assert PolygonFeed._interval_to_polygon("1h") == (1, "hour")
    assert PolygonFeed._interval_to_polygon("1d") == (1, "day")
    assert PolygonFeed._interval_to_polygon("15min") == (15, "minute")
    with pytest.raises(ValueError):
        PolygonFeed._interval_to_polygon("xyz")


def test_output_schema(monkeypatch) -> None:
    fake_client = SimpleNamespace(get_aggs=lambda *args, **kwargs: SimpleNamespace(results=_bars(), next_url=None))
    monkeypatch.setattr("data.ingestion.polygon_feed.RESTClient", lambda api_key: fake_client)
    monkeypatch.setattr("data.ingestion.polygon_feed.time.sleep", lambda *_args, **_kwargs: None)

    feed = PolygonFeed(api_key="x", rate_limit_pause=0)
    df = feed.fetch_historical("SPY", "2024-01-01", "2024-01-10", "1h")
    assert {"open", "high", "low", "close", "volume", "vwap", "transactions", "timestamp"}.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2024-01-01T09:30:00Z")
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01T09:30:00Z")


def test_rate_limit_pause(monkeypatch) -> None:
    fake_client = SimpleNamespace(get_aggs=lambda *args, **kwargs: SimpleNamespace(results=_bars(), next_url=None))
    calls = []
    monkeypatch.setattr("data.ingestion.polygon_feed.RESTClient", lambda api_key: fake_client)
    monkeypatch.setattr("data.ingestion.polygon_feed.time.sleep", lambda s: calls.append(s))

    feed = PolygonFeed(api_key="x", rate_limit_pause=12.5)
    _ = feed.fetch_historical("SPY", "2024-01-01", "2024-01-10", "1h")
    assert 12.5 in calls


def test_empty_response_raises(monkeypatch) -> None:
    fake_client = SimpleNamespace(get_aggs=lambda *args, **kwargs: SimpleNamespace(results=[], next_url=None))
    monkeypatch.setattr("data.ingestion.polygon_feed.RESTClient", lambda api_key: fake_client)
    monkeypatch.setattr("data.ingestion.polygon_feed.time.sleep", lambda *_args, **_kwargs: None)

    feed = PolygonFeed(api_key="x", rate_limit_pause=0)
    with pytest.raises(NoDataError):
        feed.fetch_historical("SPY", "2024-01-01", "2024-01-10", "1h")


def test_pagination_follows_next_url(monkeypatch) -> None:
    first = SimpleNamespace(results=_bars(5), next_url="next")
    second = SimpleNamespace(results=_bars(5), next_url=None)

    class Client:
        def __init__(self):
            self.calls = 0

        def get_aggs(self, *args, **kwargs):
            _ = args, kwargs
            self.calls += 1
            return first

        def get_next_page(self, resp):
            _ = resp
            self.calls += 1
            return second

    client = Client()
    monkeypatch.setattr("data.ingestion.polygon_feed.RESTClient", lambda api_key: client)
    monkeypatch.setattr("data.ingestion.polygon_feed.time.sleep", lambda *_args, **_kwargs: None)

    feed = PolygonFeed(api_key="x", rate_limit_pause=0)
    df = feed.fetch_historical("SPY", "2024-01-01", "2024-01-10", "1h")
    assert len(df) == 10
    assert client.calls == 2


def test_fetch_multiple_skips_failed_symbols(monkeypatch) -> None:
    def _fake_fetch(self, symbol, start, end, interval):
        _ = start, end, interval
        if symbol == "BAD":
            raise SymbolNotFoundError(symbol)
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0], "vwap": [1.0], "transactions": [1.0], "timestamp": [1]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]))

    fake_client = SimpleNamespace(get_aggs=lambda *args, **kwargs: None)
    monkeypatch.setattr("data.ingestion.polygon_feed.RESTClient", lambda api_key: fake_client)
    monkeypatch.setattr(PolygonFeed, "fetch_historical", _fake_fetch)

    feed = PolygonFeed(api_key="x", rate_limit_pause=0)
    out = feed.fetch_multiple(["SPY", "BAD", "QQQ"], "2024-01-01", "2024-01-10", "1h")
    assert set(out.keys()) == {"SPY", "QQQ"}


def test_rate_limit_retry(monkeypatch) -> None:
    class Client:
        def get_aggs(self, *args, **kwargs):
            _ = args, kwargs
            raise RuntimeError("429 Too Many Requests")

    sleeps = []
    monkeypatch.setattr("data.ingestion.polygon_feed.RESTClient", lambda api_key: Client())
    monkeypatch.setattr("data.ingestion.polygon_feed.time.sleep", lambda s: sleeps.append(s))

    feed = PolygonFeed(api_key="x", rate_limit_pause=0, max_retries=3)
    with pytest.raises(PolygonRateLimitError):
        feed.fetch_historical("SPY", "2024-01-01", "2024-01-10", "1h")
    assert sleeps.count(60) == 2


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        FeedFactory.create("polygon", config={"data": {"feed": {"rate_limit_pause": 12.5}, "polygon": {"max_retries": 3, "adjusted": True}}})


def test_auto_fallback(monkeypatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    feed = FeedFactory.create("auto", config={"data": {"feed": {"primary": "auto", "rate_limit_pause": 12.5}, "polygon": {"max_retries": 3, "adjusted": True}}})
    assert isinstance(feed, YahooFeed)

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("binance")

from execution.binance_broker import (
    BinanceBroker,
    InsufficientFundsError,
    LiveTradingNotConfiguredError,
    MinimumOrderSizeError,
)


@pytest.fixture
def paper_broker(monkeypatch) -> BinanceBroker:
    price_feed = SimpleNamespace(fetch_latest=lambda symbol, interval="1h": pd.DataFrame({"close": [50_000.0]}))
    monkeypatch.setattr("execution.binance_broker.BinanceFeed", lambda *args, **kwargs: price_feed)
    return BinanceBroker(paper_mode=True, initial_capital=10_000)


def test_paper_buy_reduces_cash(paper_broker: BinanceBroker) -> None:
    paper_broker.place_order("BTCUSDT", 0.1, "market")
    assert paper_broker.get_cash() < 10_000


def test_paper_position_tracked(paper_broker: BinanceBroker) -> None:
    paper_broker.place_order("BTCUSDT", 0.1, "market")
    assert paper_broker.get_positions()["BTCUSDT"] == pytest.approx(0.1)


def test_insufficient_funds_raises(paper_broker: BinanceBroker) -> None:
    with pytest.raises(InsufficientFundsError):
        paper_broker.place_order("BTCUSDT", 1.0, "market")


def test_minimum_order_size(paper_broker: BinanceBroker) -> None:
    with pytest.raises(MinimumOrderSizeError):
        paper_broker.place_order("BTCUSDT", 0.000001, "market")


def test_account_value_includes_positions(paper_broker: BinanceBroker) -> None:
    paper_broker.place_order("BTCUSDT", 0.1, "market")
    assert paper_broker.get_account_value() == pytest.approx(
        paper_broker.get_cash() + paper_broker.get_position_value_usd("BTCUSDT")
    )


def test_live_mode_requires_explicit_flag(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    with pytest.raises(LiveTradingNotConfiguredError):
        BinanceBroker(paper_mode=False, testnet=False)

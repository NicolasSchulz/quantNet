from __future__ import annotations

import pandas as pd
import pytest

import backtest_runner


def test_fetch_universe_cached_skips_failed_symbols(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    sample = pd.DataFrame(
        {
            "timestamp": idx,
            "open": [1, 1, 1, 1, 1],
            "high": [1, 1, 1, 1, 1],
            "low": [1, 1, 1, 1, 1],
            "close": [1, 1, 1, 1, 1],
            "volume": [1, 1, 1, 1, 1],
            "symbol": ["AAA"] * 5,
            "asset_class": ["etf"] * 5,
        },
        index=idx,
    )

    def _fake_load_or_fetch(symbol, start, end, interval, feed, store):
        _ = start, end, interval, feed, store
        if symbol == "BBB":
            raise LookupError("not found")
        return sample

    monkeypatch.setattr(backtest_runner, "_load_or_fetch_symbol", _fake_load_or_fetch)

    out = backtest_runner.fetch_universe_cached(
        symbols=["AAA", "BBB", "CCC"],
        start=idx[0],
        end=idx[-1],
        interval="1d",
        feed=None,
        store=None,
    )
    assert set(out.keys()) == {"AAA", "CCC"}


def test_require_minimum_data_raises_clear_error() -> None:
    with pytest.raises(RuntimeError, match="too few symbols"):
        backtest_runner._require_minimum_data({}, "Scenario X", min_symbols=3)

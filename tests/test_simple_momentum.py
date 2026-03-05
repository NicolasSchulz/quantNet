from __future__ import annotations

import numpy as np
import pandas as pd

from data.ingestion.yahoo_feed import YahooFeed
from strategies.examples.simple_momentum import SimpleMomentumStrategy


def _price_matrix() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=800, freq="B", tz="UTC")
    spy = np.concatenate([np.linspace(100, 150, 500), np.linspace(150, 90, 300)])
    a = np.linspace(100, 220, len(idx))
    b = np.linspace(120, 80, len(idx))
    c = np.linspace(50, 70, len(idx))
    d = np.linspace(200, 210, len(idx))
    e = np.linspace(90, 60, len(idx))
    return pd.DataFrame({"SPY": spy, "QQQ": b, "EEM": c, "GLD": d, "TLT": e, "XLK": a}, index=idx)


def test_strategy_generates_long_short_signals_without_filter() -> None:
    strat = SimpleMomentumStrategy(rebalance_freq="M", use_regime_filter=False)
    signals = strat.generate_signals(_price_matrix())
    assert not signals.empty
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert (signals == 1).any()
    assert (signals == -1).any()


def test_strategy_regime_filter_is_long_or_flat() -> None:
    strat = SimpleMomentumStrategy(
        rebalance_freq="M",
        use_regime_filter=True,
        regime_benchmark="SPY",
        tradable_symbols=["QQQ", "EEM", "GLD", "TLT", "XLK"],
    )
    signals = strat.generate_signals(_price_matrix())
    assert (signals >= 0).all()
    assert strat.regime_series is not None


def test_yahoo_feed_with_mocked_download(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    mock_df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Volume": [10, 20, 30, 40, 50],
        },
        index=idx,
    )

    def _fake_download(*args, **kwargs):
        _ = args, kwargs
        return mock_df

    monkeypatch.setattr("data.ingestion.yahoo_feed.yf.download", _fake_download)
    feed = YahooFeed()
    out = feed.fetch_historical("SPY", "2024-01-01", "2024-01-10", "1d")
    assert not out.empty
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(set(out.columns))

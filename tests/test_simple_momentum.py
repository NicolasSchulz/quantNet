from __future__ import annotations

import numpy as np
import pandas as pd

from data.ingestion.yahoo_feed import YahooFeed
from strategies.examples.simple_momentum import SimpleMomentumStrategy


def _price_matrix() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=800, freq="B", tz="UTC")
    a = np.linspace(100, 220, len(idx))
    b = np.linspace(120, 80, len(idx))
    c = np.linspace(50, 70, len(idx))
    d = np.linspace(200, 210, len(idx))
    e = np.linspace(90, 60, len(idx))
    return pd.DataFrame({"SPY": a, "QQQ": b, "EEM": c, "GLD": d, "TLT": e}, index=idx)


def test_strategy_generates_long_short_signals() -> None:
    strat = SimpleMomentumStrategy(rebalance_frequency="M")
    signals = strat.generate_signals(_price_matrix())
    assert not signals.empty
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert (signals == 1).any()
    assert (signals == -1).any()


def test_rebalance_changes_happen_monthly() -> None:
    strat = SimpleMomentumStrategy(rebalance_frequency="M")
    sig = strat.generate_signals(_price_matrix()).unstack("symbol")
    changed = sig.ne(sig.shift(1)).any(axis=1)
    changed.iloc[0] = False
    changed_dates = sig.index[changed]
    assert len(changed_dates) > 0
    assert all(d.is_month_end for d in changed_dates)


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
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(out.columns)

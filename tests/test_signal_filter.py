from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.filters.regime_filter import RegimeFilter
from strategies.ml.signal_filter import SignalFilter


def _sample_probabilities(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    probs = pd.DataFrame(
        {
            "prob_short": np.full(n, 0.2),
            "prob_flat": np.full(n, 0.2),
            "prob_long": np.full(n, 0.6),
        },
        index=idx,
    )
    return probs


def _prices(n: int = 100, bearish: bool = False) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    if bearish:
        spy = np.linspace(200, 120, n)
    else:
        spy = np.linspace(100, 180, n)
    return pd.DataFrame({"SPY": spy}, index=idx)


def test_low_confidence_filtered() -> None:
    probs = _sample_probabilities()
    probs.iloc[:10] = [0.34, 0.33, 0.33]
    sf = SignalFilter(min_confidence=0.45, regime_filter=None, min_holding_days=1)
    sig = sf.filter(probs, _prices())
    assert (sig.iloc[:10] == 0).all()


def test_regime_filter_blocks_long() -> None:
    probs = _sample_probabilities()
    rf = RegimeFilter(benchmark="SPY", ma_window=20)
    sf = SignalFilter(min_confidence=0.45, regime_filter=rf, min_holding_days=1)
    sig = sf.filter(probs, _prices(bearish=True))
    # In bearish regime, long should be blocked to flat.
    assert (sig.iloc[-20:] == 0).all()


def test_min_holding_respected() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    probs = pd.DataFrame(
        {
            "prob_short": [0.7 if i % 2 == 0 else 0.1 for i in range(20)],
            "prob_flat": [0.2] * 20,
            "prob_long": [0.1 if i % 2 == 0 else 0.7 for i in range(20)],
        },
        index=idx,
    )
    sf = SignalFilter(min_confidence=0.4, regime_filter=None, min_holding_days=3, signal_smoothing=False)
    sig = sf.filter(probs, pd.DataFrame(index=idx))
    changes = int(sig.ne(sig.shift(1)).sum() - 1)
    assert changes < 10


def test_output_only_valid_values() -> None:
    sig = SignalFilter(min_confidence=0.45).filter(_sample_probabilities(), _prices())
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_signal_stats_complete() -> None:
    sf = SignalFilter(min_confidence=0.45)
    sig = sf.filter(_sample_probabilities(), _prices())
    stats = sf.get_signal_stats(sig)
    expected = {
        "pct_long",
        "pct_short",
        "pct_flat",
        "avg_holding_days",
        "n_position_changes",
        "signals_filtered_by_confidence",
        "signals_filtered_by_regime",
    }
    assert expected.issubset(set(stats.keys()))
    total = stats["pct_long"] + stats["pct_short"] + stats["pct_flat"]
    assert abs(total - 1.0) < 1e-6

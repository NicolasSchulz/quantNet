from __future__ import annotations

import numpy as np
import pandas as pd

from risk.position_sizer import VolatilityParitySizer


def _returns() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=120, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.01, len(idx)),
            "B": rng.normal(0.0003, 0.015, len(idx)),
            "C": rng.normal(0.0002, 0.02, len(idx)),
        },
        index=idx,
    )


def test_compute_weights_respects_caps() -> None:
    sizer = VolatilityParitySizer(target_vol=0.10, lookback_days=60, max_position=0.20)
    rets = _returns()
    signals = pd.Series({"A": 1, "B": 1, "C": 1})
    weights = sizer.compute_weights(signals, rets, rets.index[-1])
    assert (weights >= 0).all()
    assert (weights <= 0.20 + 1e-12).all()


def test_compute_weights_no_active_signal() -> None:
    sizer = VolatilityParitySizer()
    rets = _returns()
    signals = pd.Series({"A": 0, "B": 0, "C": 0})
    weights = sizer.compute_weights(signals, rets, rets.index[-1])
    assert float(weights.sum()) == 0.0


def test_compare_with_equal_weight_keys() -> None:
    sizer = VolatilityParitySizer()
    rets = _returns()
    signals = pd.Series({"A": 1, "B": 1, "C": 0})
    comp = sizer.compare_with_equal_weight(signals, rets, rets.index[-1])
    assert set(comp.keys()) == {"equal_weight", "volatility_parity"}

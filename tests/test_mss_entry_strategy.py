from __future__ import annotations

import pandas as pd

from strategies.ml.mss_entry_strategy import (
    MSSEntryStrategy,
    detect_choch,
    detect_entry_candidates,
    detect_swing_points,
    track_market_structure,
)


def test_swing_points_are_confirmed_without_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [4, 5, 6, 9, 6, 5, 2, 4, 5, 4, 3, 2],
            "high": [4, 5, 6, 9, 6, 5, 2, 4, 5, 4, 3, 2],
            "low": [5, 4, 3, 2, 3, 4, 0, 4, 5, 6, 7, 8],
            "close": [4, 5, 6, 9, 6, 5, 2, 4, 5, 4, 3, 2],
            "volume": [1_000] * 12,
        },
        index=idx,
    )

    out = detect_swing_points(df, n=2)

    assert not bool(out.loc[idx[3], "is_swing_high"])
    assert bool(out.loc[idx[5], "is_swing_high"])
    assert float(out.loc[idx[5], "confirmed_swing_high"]) == 9.0

    assert not bool(out.loc[idx[6], "is_swing_low"])
    assert bool(out.loc[idx[8], "is_swing_low"])
    assert float(out.loc[idx[8], "confirmed_swing_low"]) == 0.0


def test_market_structure_choch_and_pullback_confirmation() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100, 100, 100, 95, 96],
            "high": [101, 101, 101, 101, 101, 101, 101, 101, 98, 97],
            "low": [99, 99, 99, 99, 99, 99, 99, 93, 94, 92],
            "close": [100, 100, 100, 100, 100, 100, 100, 94, 97, 93],
            "volume": [1_000] * 10,
            "is_swing_high": [False, True, False, True, False, True, False, False, False, False],
            "is_swing_low": [False, False, True, False, True, False, True, False, False, False],
            "confirmed_swing_high": [pd.NA, 100.0, pd.NA, 105.0, pd.NA, 110.0, pd.NA, pd.NA, pd.NA, pd.NA],
            "confirmed_swing_low": [pd.NA, pd.NA, 90.0, pd.NA, 92.0, pd.NA, 95.0, pd.NA, pd.NA, pd.NA],
        },
        index=idx,
    )

    structured = track_market_structure(df)
    choch = detect_choch(structured)
    entries = detect_entry_candidates(
        choch,
        pullback_timeout=5,
        min_retest_atr=0.0,
        max_overshoot_atr=5.0,
        confirmation_break_atr=0.0,
    )

    assert structured.loc[idx[6], "market_structure"] == "uptrend"
    assert bool(choch.loc[idx[7], "choch_bearish"])
    assert bool(entries.loc[idx[9], "entry_candidate_short"])
    assert not bool(entries.loc[idx[8], "entry_candidate_short"])


def test_zero_thresholds_enable_looser_pullback_confirmation() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100, 95, 98, 99],
            "high": [101, 96, 99, 100],
            "low": [99, 94, 97, 96],
            "close": [100, 98, 99, 97],
            "volume": [1_000] * 4,
            "choch_bearish": [False, True, False, False],
            "choch_bullish": [False, False, False, False],
            "last_confirmed_swing_low": [pd.NA, 100.0, 100.0, 100.0],
            "last_confirmed_swing_high": [pd.NA, pd.NA, pd.NA, pd.NA],
            "atr": [1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )

    out = detect_entry_candidates(
        df,
        pullback_timeout=5,
        min_retest_atr=0.0,
        max_overshoot_atr=5.0,
        confirmation_break_atr=0.0,
    )

    assert bool(out.loc[idx[3], "entry_candidate_short"])


def test_generate_labels_only_labels_entry_candidates() -> None:
    idx = pd.date_range("2024-01-01", periods=14, freq="h", tz="UTC")
    close = [100, 104, 101, 106, 103, 108, 105, 106, 104, 106, 103, 99, 96, 95]
    open_ = [100, 103, 102, 105, 104, 107, 106, 106, 106, 104, 106, 102, 98, 96]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    strategy = MSSEntryStrategy(
        swing_n=1,
        atr_period=2,
        tp_mult=1.0,
        sl_mult=1.0,
        max_hold=3,
        pullback_timeout=2,
        choch_min_break_atr=0.0,
        choch_min_body_fraction=0.0,
        pullback_retest_atr=0.0,
        pullback_max_overshoot_atr=5.0,
        confirmation_break_atr=0.0,
    )
    out = strategy.generate_labels(df)

    candidate_idx = out.index[out["mss_entry_candidate"]]
    assert len(candidate_idx) >= 1
    assert out.loc[~out["mss_entry_candidate"], "label"].isna().all()
    assert out.loc[candidate_idx, "entry_direction"].isin(["long", "short"]).all()
    assert (out.loc[candidate_idx, "tp_price"].notna()).all()
    assert (out.loc[candidate_idx, "sl_price"].notna()).all()
    assert -1.0 in out.loc[candidate_idx, "label"].dropna().tolist()

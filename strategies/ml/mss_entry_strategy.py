from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def detect_swing_points(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Detect confirmed swing points without leaking future information.

    A pivot at index i is only confirmed at index i+n because the rule depends on
    the next n candles. We therefore shift the boolean marker and the pivot level
    forward by n bars before returning them.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    out = df.copy()
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    swing_high = pd.Series(True, index=out.index, dtype=bool)
    swing_low = pd.Series(True, index=out.index, dtype=bool)

    for offset in range(1, n + 1):
        swing_high &= high.gt(high.shift(offset)) & high.gt(high.shift(-offset))
        swing_low &= low.lt(low.shift(offset)) & low.lt(low.shift(-offset))

    out["is_swing_high"] = swing_high.astype("boolean").shift(n, fill_value=False).astype(bool)
    out["is_swing_low"] = swing_low.astype("boolean").shift(n, fill_value=False).astype(bool)
    out["confirmed_swing_high"] = high.where(swing_high).shift(n)
    out["confirmed_swing_low"] = low.where(swing_low).shift(n)
    return out


def track_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Track HH/HL/LH/LL sequences based on confirmed swing points."""
    out = df.copy()
    market_structure = pd.Series("undefined", index=out.index, dtype="object")
    high_type = pd.Series(pd.NA, index=out.index, dtype="object")
    low_type = pd.Series(pd.NA, index=out.index, dtype="object")
    last_confirmed_high = pd.Series(np.nan, index=out.index, dtype="float64")
    last_confirmed_low = pd.Series(np.nan, index=out.index, dtype="float64")

    prev_high: float | None = None
    prev_low: float | None = None
    recent_high_types: list[str] = []
    recent_low_types: list[str] = []
    current_structure = "undefined"

    for ts in out.index:
        if bool(out.at[ts, "is_swing_high"]) and pd.notna(out.at[ts, "confirmed_swing_high"]):
            level = float(out.at[ts, "confirmed_swing_high"])
            if prev_high is not None:
                label = "HH" if level > prev_high else "LH"
                high_type.at[ts] = label
                recent_high_types.append(label)
                recent_high_types = recent_high_types[-2:]
            prev_high = level

        if bool(out.at[ts, "is_swing_low"]) and pd.notna(out.at[ts, "confirmed_swing_low"]):
            level = float(out.at[ts, "confirmed_swing_low"])
            if prev_low is not None:
                label = "HL" if level > prev_low else "LL"
                low_type.at[ts] = label
                recent_low_types.append(label)
                recent_low_types = recent_low_types[-2:]
            prev_low = level

        if recent_high_types[-2:] == ["HH", "HH"] and recent_low_types[-2:] == ["HL", "HL"]:
            current_structure = "uptrend"
        elif recent_high_types[-2:] == ["LH", "LH"] and recent_low_types[-2:] == ["LL", "LL"]:
            current_structure = "downtrend"

        market_structure.at[ts] = current_structure
        if prev_high is not None:
            last_confirmed_high.at[ts] = prev_high
        if prev_low is not None:
            last_confirmed_low.at[ts] = prev_low

    out["swing_high_type"] = high_type
    out["swing_low_type"] = low_type
    out["market_structure"] = market_structure.ffill().fillna("undefined")
    out["last_confirmed_swing_high"] = last_confirmed_high.ffill()
    out["last_confirmed_swing_low"] = last_confirmed_low.ffill()
    return out


def detect_choch(
    df: pd.DataFrame,
    min_break_atr: float = 0.0,
    min_body_fraction: float = 0.0,
) -> pd.DataFrame:
    """Detect Change of Character events from confirmed structure levels."""
    out = df.copy()
    open_ = out["open"].astype(float)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    last_high = out["last_confirmed_swing_high"].astype(float)
    last_low = out["last_confirmed_swing_low"].astype(float)
    prev_close = close.shift(1)
    atr = pd.to_numeric(out.get("atr", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
    candle_range = (high - low).clip(lower=1e-10)
    body_fraction = (close - open_).abs() / candle_range

    bear_break = (last_low - close).clip(lower=0.0)
    bull_break = (close - last_high).clip(lower=0.0)

    out["choch_bearish"] = (
        out["market_structure"].eq("uptrend")
        & last_low.notna()
        & close.lt(last_low)
        & prev_close.ge(last_low).fillna(False)
        & close.lt(open_)
        & body_fraction.ge(min_body_fraction)
        & bear_break.ge(atr * min_break_atr)
    )
    out["choch_bullish"] = (
        out["market_structure"].eq("downtrend")
        & last_high.notna()
        & close.gt(last_high)
        & prev_close.le(last_high).fillna(False)
        & close.gt(open_)
        & body_fraction.ge(min_body_fraction)
        & bull_break.ge(atr * min_break_atr)
    )
    return out


def detect_entry_candidates(
    df: pd.DataFrame,
    pullback_timeout: int = 5,
    min_retest_atr: float = 0.0,
    max_overshoot_atr: float = 1.0,
    confirmation_break_atr: float = 0.0,
) -> pd.DataFrame:
    """Detect pullback-confirmation entries after a CHoCH event."""
    if pullback_timeout <= 0:
        raise ValueError("pullback_timeout must be > 0")

    out = df.copy()
    entry_short = pd.Series(False, index=out.index, dtype=bool)
    entry_long = pd.Series(False, index=out.index, dtype=bool)

    atr = pd.to_numeric(out.get("atr", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
    short_state: dict[str, float | int | bool] | None = None
    long_state: dict[str, float | int | bool] | None = None

    for ts in out.index:
        open_price = float(out.at[ts, "open"])
        high_price = float(out.at[ts, "high"])
        low_price = float(out.at[ts, "low"])
        close_price = float(out.at[ts, "close"])
        atr_value = max(float(atr.at[ts]), 1e-10)
        is_green = close_price > open_price
        is_red = close_price < open_price

        if short_state is not None:
            short_state["bars_seen"] = int(short_state["bars_seen"]) + 1
            break_level = float(short_state["break_level"])
            if high_price > break_level + (max_overshoot_atr * atr_value):
                short_state = None
            elif is_green:
                retest_hit = min_retest_atr <= 0.0 or high_price >= break_level - (min_retest_atr * atr_value)
                if retest_hit:
                    short_state["pullback_seen"] = True
            elif bool(short_state["pullback_seen"]) and is_red and (
                confirmation_break_atr <= 0.0 or close_price < break_level - (confirmation_break_atr * atr_value)
            ):
                entry_short.at[ts] = True
                short_state = None
            elif int(short_state["bars_seen"]) >= pullback_timeout:
                short_state = None

        if long_state is not None:
            long_state["bars_seen"] = int(long_state["bars_seen"]) + 1
            break_level = float(long_state["break_level"])
            if low_price < break_level - (max_overshoot_atr * atr_value):
                long_state = None
            elif is_red:
                retest_hit = min_retest_atr <= 0.0 or low_price <= break_level + (min_retest_atr * atr_value)
                if retest_hit:
                    long_state["pullback_seen"] = True
            elif bool(long_state["pullback_seen"]) and is_green and (
                confirmation_break_atr <= 0.0 or close_price > break_level + (confirmation_break_atr * atr_value)
            ):
                entry_long.at[ts] = True
                long_state = None
            elif int(long_state["bars_seen"]) >= pullback_timeout:
                long_state = None

        # Start new search after evaluating current bar so the CHoCH candle itself
        # cannot also become the pullback confirmation candle.
        if bool(out.at[ts, "choch_bearish"]):
            short_state = {
                "bars_seen": 0,
                "pullback_seen": False,
                "break_level": float(out.at[ts, "last_confirmed_swing_low"]),
            }
        if bool(out.at[ts, "choch_bullish"]):
            long_state = {
                "bars_seen": 0,
                "pullback_seen": False,
                "break_level": float(out.at[ts, "last_confirmed_swing_high"]),
            }

    out["entry_candidate_short"] = entry_short
    out["entry_candidate_long"] = entry_long
    return out


@dataclass
class MSSEntryStrategy:
    swing_n: int = 4
    atr_period: int = 14
    tp_mult: float = 3.0
    sl_mult: float = 1.5
    max_hold: int = 48
    pullback_timeout: int = 5
    choch_min_break_atr: float = 0.15
    choch_min_body_fraction: float = 0.35
    pullback_retest_atr: float = 0.25
    pullback_max_overshoot_atr: float = 0.35
    confirmation_break_atr: float = 0.05

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input index must be a pandas DatetimeIndex.")
        if self.swing_n <= 0:
            raise ValueError("swing_n must be > 0")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be > 0")
        if self.max_hold <= 0:
            raise ValueError("max_hold must be > 0")

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(self.atr_period, min_periods=self.atr_period).mean()

    def apply_mss_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        out = detect_swing_points(df, n=self.swing_n)
        out["atr"] = self._atr(out)
        out = track_market_structure(out)
        out = detect_choch(
            out,
            min_break_atr=self.choch_min_break_atr,
            min_body_fraction=self.choch_min_body_fraction,
        )
        out = detect_entry_candidates(
            out,
            pullback_timeout=self.pullback_timeout,
            min_retest_atr=self.pullback_retest_atr,
            max_overshoot_atr=self.pullback_max_overshoot_atr,
            confirmation_break_atr=self.confirmation_break_atr,
        )
        out["mss_entry_candidate"] = out["entry_candidate_short"] | out["entry_candidate_long"]
        out["entry_direction"] = pd.Series(pd.NA, index=out.index, dtype="object")
        out.loc[out["entry_candidate_long"], "entry_direction"] = "long"
        out.loc[out["entry_candidate_short"], "entry_direction"] = "short"
        out["entry_direction_bias"] = (
            out["entry_direction"]
            .map({"short": -1.0, "long": 1.0})
            .fillna(0.0)
            .astype("float64")
        )
        return out

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MSS-filtered entry candidates plus ATR-based triple-barrier labels."""
        self._validate_input(df)

        out = self.apply_mss_filter(df)
        out["tp_price"] = np.nan
        out["sl_price"] = np.nan
        out["time_barrier"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        out["label"] = np.nan

        high = out["high"].astype(float)
        low = out["low"].astype(float)
        close = out["close"].astype(float)
        valid_candidates = out.index[out["mss_entry_candidate"]]

        for ts in valid_candidates:
            i = out.index.get_loc(ts)
            if isinstance(i, slice):
                continue
            if i + self.max_hold >= len(out.index):
                continue

            entry_price = float(close.iloc[i])
            atr_value = float(out["atr"].iloc[i]) if pd.notna(out["atr"].iloc[i]) else np.nan
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue

            direction = out.at[ts, "entry_direction"]
            if direction == "long":
                tp_price = entry_price + (self.tp_mult * atr_value)
                sl_price = entry_price - (self.sl_mult * atr_value)
            else:
                tp_price = entry_price - (self.tp_mult * atr_value)
                sl_price = entry_price + (self.sl_mult * atr_value)

            assigned = 0.0
            realized_ret = 0.0
            time_barrier = out.index[i + self.max_hold]

            for j in range(i + 1, i + self.max_hold + 1):
                if high.iloc[j] >= max(tp_price, sl_price):
                    assigned = 1.0
                    realized_ret = (max(tp_price, sl_price) / entry_price) - 1.0
                    break
                if low.iloc[j] <= min(tp_price, sl_price):
                    assigned = -1.0
                    realized_ret = (min(tp_price, sl_price) / entry_price) - 1.0
                    break
                if j == i + self.max_hold:
                    assigned = 0.0
                    realized_ret = (close.iloc[j] / entry_price) - 1.0

            out.at[ts, "tp_price"] = tp_price
            out.at[ts, "sl_price"] = sl_price
            out.at[ts, "time_barrier"] = time_barrier
            out.at[ts, "label"] = assigned if abs(realized_ret) >= 0.0 else 0.0

        candidate_count = int(out["mss_entry_candidate"].sum())
        candidate_pct = (candidate_count / max(len(out), 1)) * 100.0
        label_distribution = out.loc[out["mss_entry_candidate"], "label"].dropna().value_counts().to_dict()
        LOGGER.info(
            "MSS entry candidates: %d/%d (%.2f%%). Label distribution: %s",
            candidate_count,
            len(out),
            candidate_pct,
            label_distribution,
        )
        return out

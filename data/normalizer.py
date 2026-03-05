from __future__ import annotations

import logging
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

LOGGER = logging.getLogger(__name__)

SUPPORTED_ASSET_CLASSES: set[str] = {"equity", "crypto", "etf"}
SCHEMA_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "symbol",
    "asset_class",
]


def _normalize_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Yahoo columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df.index, utc=True),
            "open": pd.to_numeric(df["Open"], errors="coerce"),
            "high": pd.to_numeric(df["High"], errors="coerce"),
            "low": pd.to_numeric(df["Low"], errors="coerce"),
            "close": pd.to_numeric(df["Close"], errors="coerce"),
            "volume": pd.to_numeric(df["Volume"], errors="coerce"),
            "vwap": pd.NA,
        }
    )
    return out


def _normalize_polygon(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Polygon columns: {sorted(missing)}")

    if "timestamp" in df.columns:
        raw_ts = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(raw_ts):
            ts = pd.to_datetime(raw_ts, utc=True)
        elif pd.api.types.is_numeric_dtype(raw_ts):
            # Polygon aggregate timestamps are epoch milliseconds.
            ts = pd.to_datetime(raw_ts, unit="ms", utc=True)
        else:
            ts = pd.to_datetime(raw_ts, utc=True)
    elif "t" in df.columns:
        ts = pd.to_datetime(df["t"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df.index, utc=True)

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
            "vwap": pd.to_numeric(df.get("vwap", pd.NA), errors="coerce"),
        }
    )
    return out


def check_gaps(df: pd.DataFrame, interval: str, warn_threshold: float = 0.05) -> None:
    if df.empty:
        return
    if interval not in {"1h", "1d", "15min", "5min"}:
        return

    freq_map = {"1h": "1h", "1d": "1D", "15min": "15min", "5min": "5min"}
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq_map[interval], tz="UTC")
    expected = len(full)
    actual = len(df.index.unique())
    if expected <= 0:
        return

    gap_ratio = max(0.0, (expected - actual) / expected)
    if gap_ratio > warn_threshold:
        LOGGER.warning(
            "Detected %.2f%% timestamp gaps for interval %s (%d actual vs %d expected).",
            gap_ratio * 100,
            interval,
            actual,
            expected,
        )


def _validate_trading_hours(df: pd.DataFrame) -> None:
    if df.empty:
        return
    ny = ZoneInfo("America/New_York")
    local_ts = df.index.tz_convert(ny)
    start = time(hour=9, minute=30)
    end = time(hour=16, minute=0)

    outside = [t for t in local_ts if t.time() < start or t.time() > end]
    if outside:
        LOGGER.warning("Found %d bars outside regular trading hours (ET).", len(outside))


def normalize_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    validate_trading_hours: bool = False,
    interval: str | None = None,
) -> pd.DataFrame:
    """Normalize provider OHLCV output to unified schema.

    Output schema:
    [timestamp, open, high, low, close, volume, vwap, symbol, asset_class]

    - All timestamps are UTC.
    - VWAP is optional and NaN for feeds that do not provide it.
    """
    if asset_class not in SUPPORTED_ASSET_CLASSES:
        raise ValueError(
            f"Unsupported asset_class '{asset_class}'. Supported: {sorted(SUPPORTED_ASSET_CLASSES)}"
        )
    if df is None or df.empty:
        raise ValueError("Cannot normalize empty DataFrame.")

    lower = {c.lower() for c in df.columns}
    if {"open", "high", "low", "close", "volume"}.issubset(lower):
        # Handle already-lowercase or polygon native schema.
        mapped = df.copy()
        mapped.columns = [c.lower() for c in mapped.columns]
        normalized = _normalize_polygon(mapped)
    else:
        normalized = _normalize_yahoo(df)

    normalized["symbol"] = symbol.upper()
    normalized["asset_class"] = asset_class

    if normalized[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC columns contain NaN values after normalization.")
    if normalized["volume"].isna().any() or (normalized["volume"] < 0).any():
        raise ValueError("Volume must be non-negative and non-null.")

    normalized = normalized[SCHEMA_COLUMNS]
    normalized.index = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized.index.name = "timestamp"

    if validate_trading_hours:
        _validate_trading_hours(normalized)
    if interval is not None:
        check_gaps(normalized, interval=interval)

    return normalized

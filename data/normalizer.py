from __future__ import annotations

from typing import Final

import pandas as pd

SUPPORTED_ASSET_CLASSES: Final[set[str]] = {"equity", "crypto", "etf"}
SCHEMA_COLUMNS: Final[list[str]] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "asset_class",
]


def normalize_ohlcv(df: pd.DataFrame, symbol: str, asset_class: str) -> pd.DataFrame:
    """Normalize provider OHLCV output into the platform schema.

    Output uses UTC DatetimeIndex named ``timestamp`` and keeps a ``timestamp``
    column for explicit tabular workflows.
    """
    if asset_class not in SUPPORTED_ASSET_CLASSES:
        raise ValueError(
            f"Unsupported asset_class '{asset_class}'. Supported: {sorted(SUPPORTED_ASSET_CLASSES)}"
        )

    if df is None or df.empty:
        raise ValueError("Cannot normalize empty DataFrame.")

    required_source_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_source_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")

    index = pd.to_datetime(df.index, utc=True)

    normalized = pd.DataFrame(
        {
            "timestamp": index,
            "open": pd.to_numeric(df["Open"], errors="coerce"),
            "high": pd.to_numeric(df["High"], errors="coerce"),
            "low": pd.to_numeric(df["Low"], errors="coerce"),
            "close": pd.to_numeric(df["Close"], errors="coerce"),
            "volume": pd.to_numeric(df["Volume"], errors="coerce"),
            "symbol": symbol.upper(),
            "asset_class": asset_class,
        }
    )

    if normalized[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC columns contain NaN values after normalization.")

    if normalized["volume"].isna().any() or (normalized["volume"] < 0).any():
        raise ValueError("Volume must be non-negative and non-null.")

    normalized = normalized[SCHEMA_COLUMNS]
    normalized.index = normalized["timestamp"]
    normalized.index.name = "timestamp"
    return normalized

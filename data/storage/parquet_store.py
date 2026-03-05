from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    """Local parquet cache for normalized OHLCV data."""

    def __init__(self, storage_path: str) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _file_path(self, symbol: str, interval: str) -> Path:
        safe_symbol = symbol.upper().replace("/", "-")
        return self.storage_path / f"{safe_symbol}__{interval}.parquet"

    def save(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        if df.empty:
            raise ValueError("Cannot save empty DataFrame to parquet.")
        file_path = self._file_path(symbol=symbol, interval=interval)
        df.to_parquet(file_path, index=True)
        return file_path

    def load(
        self,
        symbol: str,
        interval: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        file_path = self._file_path(symbol=symbol, interval=interval)
        if not file_path.exists():
            raise FileNotFoundError(f"No parquet cache found for {symbol} @ {interval}.")

        df = pd.read_parquet(file_path)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            df.index = ts
            df.index.name = "timestamp"
        else:
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"

        def _to_utc(ts_like: str | pd.Timestamp) -> pd.Timestamp:
            ts = pd.Timestamp(ts_like)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")

        if start is not None:
            start_ts = _to_utc(start)
            df = df[df.index >= start_ts]
        if end is not None:
            end_ts = _to_utc(end)
            df = df[df.index <= end_ts]

        return df.sort_index()

    def exists(self, symbol: str, interval: str) -> bool:
        return self._file_path(symbol=symbol, interval=interval).exists()

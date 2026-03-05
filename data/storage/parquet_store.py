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

        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
            df = df[df.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
            df = df[df.index <= end_ts]

        return df.sort_index()

    def exists(self, symbol: str, interval: str) -> bool:
        return self._file_path(symbol=symbol, interval=interval).exists()

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd


class ParquetStore:
    """Local parquet cache for normalized OHLCV data."""

    def __init__(self, storage_path: str) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _dir_path(self, symbol: str, interval: str = "1h") -> Path:
        safe_symbol = symbol.upper().replace("/", "-")
        return self.storage_path / safe_symbol / interval

    def _file_path(self, symbol: str, interval: str = "1h") -> Path:
        return self._dir_path(symbol=symbol, interval=interval) / "data.parquet"

    def save(self, df: pd.DataFrame, symbol: str, interval: str = "1h") -> Path:
        if df.empty:
            raise ValueError("Cannot save empty DataFrame to parquet.")
        file_path = self._file_path(symbol=symbol, interval=interval)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, index=True)
        return file_path

    def append(self, symbol: str, interval: str, new_data: pd.DataFrame) -> None:
        if new_data.empty:
            return
        if self.exists(symbol, interval):
            old = self.load(symbol, interval)
            combined = pd.concat([old, new_data], axis=0)
        else:
            combined = new_data.copy()

        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self.save(combined, symbol, interval)

    def _to_utc(self, ts_like: str | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts_like)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def load(
        self,
        symbol: str,
        interval: str = "1h",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        file_path = self._file_path(symbol=symbol, interval=interval)
        if not file_path.exists():
            raise FileNotFoundError(f"No parquet cache found for {symbol} @ {interval}.")

        df = pd.read_parquet(file_path)
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], utc=True)
        else:
            df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"

        if start is not None:
            df = df[df.index >= self._to_utc(start)]
        if end is not None:
            df = df[df.index <= self._to_utc(end)]

        return df.sort_index()

    def exists(self, symbol: str, interval: str = "1h") -> bool:
        return self._file_path(symbol=symbol, interval=interval).exists()

    def get_date_range(self, symbol: str, interval: str = "1h") -> tuple[pd.Timestamp, pd.Timestamp]:
        df = self.load(symbol, interval)
        if df.empty:
            raise ValueError(f"No data in cache for {symbol}@{interval}")
        return df.index.min(), df.index.max()

    def needs_update(self, symbol: str, interval: str = "1h", max_age_hours: int = 4) -> bool:
        if not self.exists(symbol, interval):
            return True
        _, end = self.get_date_range(symbol, interval)
        age = pd.Timestamp.now(tz=timezone.utc) - end
        return age > pd.Timedelta(hours=max_age_hours)

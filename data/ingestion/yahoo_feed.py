from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from data.ingestion.base_feed import BaseFeed


@dataclass
class YahooFeed(BaseFeed):
    """Yahoo Finance feed implementation via yfinance."""

    supported_intervals: tuple[str, ...] = ("1d", "1h", "1wk")

    def _validate_interval(self, interval: str) -> None:
        if interval not in self.supported_intervals:
            raise ValueError(
                f"Unsupported interval '{interval}'. Supported: {self.supported_intervals}"
            )

    def _download(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str,
    ) -> pd.DataFrame:
        self._validate_interval(interval)
        try:
            df = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:  # pragma: no cover - provider/network dependent
            raise RuntimeError(f"Yahoo Finance request failed for {symbol}: {exc}") from exc

        if df is None or df.empty:
            raise LookupError(f"No data found for symbol '{symbol}' in requested range.")

        # Flatten possible multi-index columns when yfinance returns them.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise LookupError(f"Yahoo data for '{symbol}' missing columns: {sorted(missing)}")

        return df

    def fetch_historical(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from Yahoo Finance."""
        return self._download(symbol=symbol.upper(), start=start, end=end, interval=interval)

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """Fetch latest available bar by requesting a short lookback window."""
        now = pd.Timestamp.utcnow()
        # Use daily bars for robust latest lookup.
        start = now - pd.Timedelta(days=10)
        df = self._download(symbol=symbol.upper(), start=start, end=now, interval="1d")
        latest = df.tail(1)
        if latest.empty:
            raise LookupError(f"No latest data available for symbol '{symbol}'.")
        return latest

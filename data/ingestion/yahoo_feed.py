from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from data.ingestion.base_feed import BaseFeed


@dataclass
class YahooFeed(BaseFeed):
    """Yahoo Finance feed implementation via yfinance."""

    supported_intervals: tuple[str, ...] = ("1d", "1h", "1wk")
    crypto_aliases: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.crypto_aliases is None:
            self.crypto_aliases = {
                "BTC": "BTC-USD",
                "ETH": "ETH-USD",
                "SOL": "SOL-USD",
                "XRP": "XRP-USD",
                "ADA": "ADA-USD",
                "DOGE": "DOGE-USD",
                "BNB": "BNB-USD",
            }

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

    def _symbol_candidates(self, symbol: str) -> list[str]:
        s = symbol.strip().upper()
        candidates = [s]

        # Common crypto shorthand fallback.
        mapped = self.crypto_aliases.get(s)
        if mapped and mapped not in candidates:
            candidates.append(mapped)

        # Heuristic: raw crypto ticker might require USD quote on Yahoo.
        if "-" not in s and s in {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB"}:
            usd_pair = f"{s}-USD"
            if usd_pair not in candidates:
                candidates.append(usd_pair)

        return candidates

    def fetch_historical(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from Yahoo Finance."""
        errors: list[str] = []
        for candidate in self._symbol_candidates(symbol):
            try:
                return self._download(symbol=candidate, start=start, end=end, interval=interval)
            except LookupError as exc:
                errors.append(f"{candidate}: {exc}")
                continue

        raise LookupError(
            f"No data found for '{symbol}' in requested range. "
            f"Tried symbols: {self._symbol_candidates(symbol)}. "
            f"Details: {' | '.join(errors[:3])}"
        )

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """Fetch latest available bar by requesting a short lookback window."""
        now = pd.Timestamp.utcnow()
        # Use daily bars for robust latest lookup.
        start = now - pd.Timedelta(days=10)
        df = self.fetch_historical(symbol=symbol, start=start, end=now, interval="1d")
        latest = df.tail(1)
        if latest.empty:
            raise LookupError(f"No latest data available for symbol '{symbol}'.")
        return latest

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeed(ABC):
    """Abstract interface for market data providers.

    Implementations must return a pandas DataFrame indexed by timestamp.
    Expected raw columns from a feed are OHLCV-like fields (Open, High, Low,
    Close, Volume), which are normalized later via data.normalizer.
    """

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch historical bars for a symbol.

        Args:
            symbol: Instrument ticker (e.g., SPY, BTC-USD).
            start: Inclusive start datetime/date.
            end: Exclusive or provider-specific end datetime/date.
            interval: Bar interval such as '1d', '1h', '1wk'.

        Returns:
            DataFrame indexed by timestamps with provider raw OHLCV columns.

        Raises:
            ValueError: Unsupported parameters.
            LookupError: Symbol/data not found.
            RuntimeError: Provider or transport-level failure.
        """

    @abstractmethod
    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """Fetch latest available bar(s) for a symbol.

        Args:
            symbol: Instrument ticker.

        Returns:
            DataFrame indexed by timestamps containing the latest bar data.

        Raises:
            LookupError: No data available for symbol.
            RuntimeError: Provider failure.
        """

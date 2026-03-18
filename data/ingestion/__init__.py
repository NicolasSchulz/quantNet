"""Market data feed interfaces and implementations."""

from data.ingestion.base_feed import BaseFeed
from data.ingestion.feed_factory import FeedFactory, MissingApiKeyError
from data.ingestion.yahoo_feed import YahooFeed
try:  # pragma: no cover - optional dependency
    from data.ingestion.binance_feed import BinanceFeed, NoDataError as BinanceNoDataError, SymbolNotFoundError as BinanceSymbolNotFoundError
except Exception:  # pragma: no cover
    BinanceFeed = None  # type: ignore[assignment]
    BinanceNoDataError = LookupError  # type: ignore[assignment]
    BinanceSymbolNotFoundError = LookupError  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from data.ingestion.polygon_feed import (
        NoDataError,
        PolygonAuthError,
        PolygonFeed,
        PolygonRateLimitError,
        SymbolNotFoundError,
    )
except Exception:  # pragma: no cover
    PolygonFeed = None  # type: ignore[assignment]
    PolygonRateLimitError = RuntimeError  # type: ignore[assignment]
    PolygonAuthError = RuntimeError  # type: ignore[assignment]
    SymbolNotFoundError = LookupError  # type: ignore[assignment]
    NoDataError = LookupError  # type: ignore[assignment]

__all__ = [
    "BaseFeed",
    "FeedFactory",
    "MissingApiKeyError",
    "PolygonFeed",
    "PolygonRateLimitError",
    "PolygonAuthError",
    "SymbolNotFoundError",
    "NoDataError",
    "YahooFeed",
    "BinanceFeed",
    "BinanceNoDataError",
    "BinanceSymbolNotFoundError",
]

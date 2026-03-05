from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from data.ingestion.base_feed import BaseFeed

try:
    from polygon import RESTClient
except ImportError as exc:  # pragma: no cover - dependency/environment specific
    raise ImportError("polygon-api-client is required for PolygonFeed") from exc

LOGGER = logging.getLogger(__name__)


class PolygonRateLimitError(RuntimeError):
    pass


class PolygonAuthError(RuntimeError):
    pass


class SymbolNotFoundError(LookupError):
    pass


class NoDataError(LookupError):
    pass


@dataclass
class PolygonFeed(BaseFeed):
    """Polygon.io REST feed.

    Notes:
        1h data contains bars only during market hours and naturally skips
        weekends/off-hours. Gaps outside trading windows are expected.
    """

    api_key: str | None = None
    rate_limit_pause: float = 12.5
    max_retries: int = 3
    adjusted: bool = True

    def __post_init__(self) -> None:
        load_dotenv("config/secrets.env")
        if self.api_key is None:
            import os

            self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise PolygonAuthError(
                "POLYGON_API_KEY missing. Add it to config/secrets.env."
            )
        self.client = RESTClient(api_key=self.api_key)
        LOGGER.info("PolygonFeed initialized. Rate limit pause: %.1fs", self.rate_limit_pause)

    @staticmethod
    def _interval_to_polygon(interval: str) -> tuple[int, str]:
        mapping = {
            "1h": (1, "hour"),
            "1d": (1, "day"),
            "15min": (15, "minute"),
            "5min": (5, "minute"),
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval '{interval}'. Supported: {sorted(mapping)}")
        return mapping[interval]

    def _safe_call(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        for attempt in range(1, self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                time.sleep(self.rate_limit_pause)
                return result
            except Exception as exc:  # pragma: no cover - external API behavior
                msg = str(exc).lower()
                if "403" in msg or "forbidden" in msg or "auth" in msg:
                    raise PolygonAuthError(
                        "Polygon auth failed (403). Check POLYGON_API_KEY in config/secrets.env."
                    ) from exc
                if "429" in msg or "rate" in msg:
                    if attempt >= self.max_retries:
                        raise PolygonRateLimitError("Polygon rate limit retry exhausted") from exc
                    LOGGER.warning("Polygon 429 rate limit. Retry %d/%d in 60s", attempt, self.max_retries)
                    time.sleep(60)
                    continue

                if attempt >= self.max_retries:
                    raise RuntimeError(f"Polygon request failed after retries: {exc}") from exc
                backoff = 2 ** attempt
                LOGGER.warning(
                    "Polygon request failed (%s). Retry %d/%d in %ss",
                    exc,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError("Unreachable Polygon retry state")

    @staticmethod
    def _extract_results(response: Any) -> tuple[list[Any], str | None]:
        if response is None:
            return [], None

        if isinstance(response, list):
            return response, None

        if hasattr(response, "results"):
            results = list(getattr(response, "results") or [])
            next_url = getattr(response, "next_url", None)
            return results, next_url

        # Some client versions may return iterator-like values.
        try:
            as_list = list(response)
            return as_list, None
        except TypeError:
            return [], None

    @staticmethod
    def _bars_to_frame(results: list[Any]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for bar in results:
            timestamp_ms = int(getattr(bar, "t", getattr(bar, "timestamp", 0)))
            timestamp = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
            # Polygon Agg fields: o,h,l,c,v,vw,t,n
            rows.append(
                {
                    "open": float(getattr(bar, "o", getattr(bar, "open", float("nan")))),
                    "high": float(getattr(bar, "h", getattr(bar, "high", float("nan")))),
                    "low": float(getattr(bar, "l", getattr(bar, "low", float("nan")))),
                    "close": float(getattr(bar, "c", getattr(bar, "close", float("nan")))),
                    "volume": float(getattr(bar, "v", getattr(bar, "volume", float("nan")))),
                    "vwap": float(getattr(bar, "vw", getattr(bar, "vwap", float("nan")))),
                    "transactions": float(getattr(bar, "n", getattr(bar, "transactions", float("nan")))),
                    "timestamp": timestamp,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.index = pd.DatetimeIndex(df["timestamp"])
        df.index.name = "timestamp"
        return df.sort_index()

    def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        multiplier, timespan = self._interval_to_polygon(interval)

        response = self._safe_call(
            self.client.get_aggs,
            symbol.upper(),
            multiplier,
            timespan,
            start,
            end,
            limit=50000,
            adjusted=self.adjusted,
        )
        results, next_url = self._extract_results(response)

        while next_url:
            LOGGER.info("Polygon pagination for %s (%s)", symbol.upper(), interval)
            response = self._safe_call(self.client.get_next_page, response) if hasattr(self.client, "get_next_page") else self._safe_call(self.client.get_aggs, next_url=next_url)
            page_results, next_url = self._extract_results(response)
            results.extend(page_results)

        if not results:
            raise NoDataError(
                f"No data for symbol '{symbol.upper()}' in {start}..{end} @ {interval}"
            )

        df = self._bars_to_frame(results)
        if df.empty:
            raise NoDataError(
                f"Empty data for symbol '{symbol.upper()}' in {start}..{end} @ {interval}"
            )
        return df

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        end = pd.Timestamp.utcnow()
        start = end - timedelta(days=7)
        return self.fetch_historical(
            symbol=symbol,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval="1h",
        )

    def fetch_multiple(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1h",
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        total = len(symbols)
        for i, symbol in enumerate(symbols, start=1):
            LOGGER.info("Lade Symbol %d/%d: %s (warte %.1fs...)", i, total, symbol, self.rate_limit_pause)
            try:
                out[symbol.upper()] = self.fetch_historical(symbol, start, end, interval)
            except Exception as exc:
                LOGGER.warning("Symbol %s fehlgeschlagen: %s", symbol, exc)
        return out

    def validate_api_key(self) -> bool:
        try:
            end = pd.Timestamp.utcnow().date().isoformat()
            start = (pd.Timestamp.utcnow() - pd.Timedelta(days=2)).date().isoformat()
            _ = self.fetch_historical("AAPL", start=start, end=end, interval="1d")
            return True
        except PolygonAuthError:
            return False
        except Exception:
            return True

    def get_market_status(self) -> dict[str, str]:
        try:
            if hasattr(self.client, "get_market_status"):
                status = self._safe_call(self.client.get_market_status)
            else:
                status = self._safe_call(self.client.get_market_status_now)
        except Exception as exc:  # pragma: no cover - external API behavior
            raise RuntimeError(f"Failed to fetch market status: {exc}") from exc

        market = str(getattr(status, "market", getattr(status, "market_status", "unknown")))
        server_time = str(getattr(status, "serverTime", getattr(status, "server_time", "")))
        return {"market": market, "serverTime": server_time}

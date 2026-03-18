from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from data.ingestion.base_feed import BaseFeed

try:
    from binance import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except ImportError as exc:  # pragma: no cover - dependency/environment specific
    raise ImportError("python-binance is required for BinanceFeed") from exc

LOGGER = logging.getLogger(__name__)


class SymbolNotFoundError(LookupError):
    pass


class NoDataError(LookupError):
    def __init__(self, symbol: str, start: str | None, end: str | None, interval: str) -> None:
        message = f"No data for symbol '{symbol}'"
        if start is not None and end is not None:
            message += f" in {start}..{end}"
        message += f" @ {interval}"
        super().__init__(message)


@dataclass
class BinanceFeed(BaseFeed):
    """Binance market data adapter.

    Crypto-Daten haben keine Wochenend-Luecken; die Backtesting-Engine darf
    daher bei `asset_class="crypto"` keine Marktstundenfilter anwenden.
    """

    api_key: str | None = None
    api_secret: str | None = None
    testnet: bool = False
    max_retries: int = 3

    def __post_init__(self) -> None:
        load_dotenv("config/secrets.env")
        if self.api_key is None:
            self.api_key = os.getenv("BINANCE_API_KEY")
        if self.api_secret is None:
            self.api_secret = os.getenv("BINANCE_API_SECRET")

        self.client = Client(api_key=self.api_key, api_secret=self.api_secret, ping=False)
        if self.testnet:
            self.client.API_URL = "https://testnet.binance.vision/api"

    @staticmethod
    def _interval_to_binance(interval: str) -> str:
        mapping = {
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "15min": Client.KLINE_INTERVAL_15MINUTE,
            "4h": Client.KLINE_INTERVAL_4HOUR,
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval '{interval}'. Supported: {sorted(mapping)}")
        return mapping[interval]

    @staticmethod
    def _bars_to_frame(bars: list[list[Any]]) -> pd.DataFrame:
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(bars, columns=columns)
        if df.empty:
            return df

        keep = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "number_of_trades",
            "taker_buy_base_volume",
        ]
        out = df[keep].copy()
        for col in keep[1:]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out.index = pd.to_datetime(out["open_time"], unit="ms", utc=True)
        out.index.name = "timestamp"
        return out.sort_index()

    def _safe_call(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except BinanceAPIException as exc:
                if exc.code == -1121 or "invalid symbol" in str(exc).lower():
                    raise SymbolNotFoundError(kwargs.get("symbol") or (args[0] if args else "UNKNOWN")) from exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Binance API request failed after retries: {exc}") from exc
            except (BinanceRequestException, OSError, ConnectionError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Binance request failed after retries: {exc}") from exc
            backoff = 2 ** (attempt - 1)
            LOGGER.warning("Binance request failed. Retry %d/%d in %ss", attempt, self.max_retries, backoff)
            time.sleep(backoff)

        raise RuntimeError("Unreachable Binance retry state")

    def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch raw Binance klines with automatic pagination.

        Crypto-Daten handeln 24/7 ohne Wochenend-Luecken. Die Backtesting-
        Engine muss diese kontinuierlichen Bars unveraendert verarbeiten.
        """
        interval_code = self._interval_to_binance(interval)
        symbol = symbol.upper()
        current_start: str | int = start
        all_bars: list[list[Any]] = []

        while True:
            bars = self._safe_call(
                self.client.get_historical_klines,
                symbol=symbol,
                interval=interval_code,
                start_str=current_start,
                end_str=end,
                limit=1000,
            )
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < 1000:
                break
            current_start = int(bars[-1][0]) + 1
            time.sleep(0.1)

        if not all_bars:
            raise NoDataError(symbol, start, end, interval)

        df = self._bars_to_frame(all_bars)
        if df.empty:
            raise NoDataError(symbol, start, end, interval)
        return df

    def fetch_latest(self, symbol: str, interval: str = "1h") -> pd.DataFrame:
        interval_code = self._interval_to_binance(interval)
        bars = self._safe_call(self.client.get_klines, symbol=symbol.upper(), interval=interval_code, limit=200)
        if not bars:
            raise NoDataError(symbol.upper(), None, None, interval)
        return self._bars_to_frame(bars)

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
            LOGGER.info("Lade %s (%d/%d)...", symbol.upper(), i, total)
            try:
                out[symbol.upper()] = self.fetch_historical(symbol, start, end, interval)
            except Exception as exc:
                LOGGER.warning("Binance fetch fuer %s uebersprungen: %s", symbol.upper(), exc)
            time.sleep(0.1)
        return out

    def validate_connection(self) -> bool:
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_server_time(self) -> datetime:
        payload = self._safe_call(self.client.get_server_time)
        server_time_ms = int(payload["serverTime"])
        return pd.to_datetime(server_time_ms, unit="ms", utc=True).to_pydatetime()

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from uuid import uuid4

from dotenv import load_dotenv

from backtesting.cost_model import CostModel
from data.ingestion.binance_feed import BinanceFeed
from execution.base_broker import BaseBroker, Order

try:
    from binance import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except ImportError as exc:  # pragma: no cover - dependency/environment specific
    raise ImportError("python-binance is required for BinanceBroker") from exc

LOGGER = logging.getLogger(__name__)

MIN_ORDER_SIZES = {
    "BTCUSDT": 0.00001,
    "ETHUSDT": 0.0001,
    "SOLUSDT": 0.01,
}


class InsufficientFundsError(RuntimeError):
    pass


class MinimumOrderSizeError(ValueError):
    pass


class LiveTradingNotConfiguredError(RuntimeError):
    pass


@dataclass
class BinanceBroker(BaseBroker):
    api_key: str | None = None
    api_secret: str | None = None
    testnet: bool = True
    paper_mode: bool = True
    initial_capital: float = 100_000.0
    cost_model: CostModel = field(default_factory=lambda: CostModel(asset_class="crypto"))
    price_feed: BinanceFeed | None = None
    max_retries: int = 3

    def __post_init__(self) -> None:
        load_dotenv("config/secrets.env")
        if self.api_key is None:
            self.api_key = os.getenv("BINANCE_API_KEY")
        if self.api_secret is None:
            self.api_secret = os.getenv("BINANCE_API_SECRET")

        self.cash = float(self.initial_capital)
        self.positions: dict[str, float] = {}
        self.orders: list[Order] = []
        self.pending_orders: dict[str, Order] = {}

        if self.paper_mode:
            self.price_feed = self.price_feed or BinanceFeed(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )
            self.client = None
            return

        if not self.testnet and (not self.api_key or not self.api_secret):
            raise LiveTradingNotConfiguredError(
                "Live trading requires explicit API credentials in config/secrets.env."
            )

        self.price_feed = self.price_feed or BinanceFeed(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret, ping=False)
        if self.testnet:
            self.client.API_URL = "https://testnet.binance.vision/api"
        self.client.ping()
        if not self.testnet:
            LOGGER.warning("LIVE TRADING AKTIV - echte Orders werden platziert")

    def _ensure_min_size(self, symbol: str, quantity: float) -> None:
        minimum = MIN_ORDER_SIZES.get(symbol.upper())
        if minimum is not None and abs(quantity) < minimum:
            raise MinimumOrderSizeError(
                f"{symbol.upper()} requires at least {minimum} base units per order."
            )

    def _latest_price(self, symbol: str) -> float:
        if self.price_feed is None:
            raise RuntimeError("Price feed not configured.")
        latest = self.price_feed.fetch_latest(symbol=symbol, interval="1h")
        return float(latest["close"].iloc[-1])

    def _safe_live_call(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (BinanceAPIException, BinanceRequestException, OSError, ConnectionError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Binance order request failed after retries: {exc}") from exc
                backoff = 2 ** (attempt - 1)
                LOGGER.warning("Binance order request failed. Retry %d/%d in %ss", attempt, self.max_retries, backoff)
                time.sleep(backoff)
        raise RuntimeError("Unreachable Binance order retry state")

    def place_order(
        self,
        symbol: str,
        quantity: float,
        order_type: str,
        price: float | None = None,
    ) -> Order:
        symbol = symbol.upper()
        normalized_type = order_type.lower()
        if normalized_type not in {"market", "limit"}:
            raise ValueError("order_type must be 'market' or 'limit'")
        if quantity == 0:
            raise ValueError("quantity cannot be zero")

        self._ensure_min_size(symbol, quantity)

        if self.paper_mode:
            reference_price = float(price) if price is not None else self._latest_price(symbol)
            side = 1 if quantity > 0 else -1
            fill_price = self.cost_model.apply(reference_price, abs(quantity), side)
            notional = abs(quantity) * fill_price
            commission = self.cost_model.calculate_commission(notional)

            if quantity > 0 and (self.cash < notional + commission):
                raise InsufficientFundsError(
                    f"Insufficient cash for {symbol}: need {notional + commission:.2f}, have {self.cash:.2f}"
                )

            order = Order(
                id=str(uuid4()),
                symbol=symbol,
                quantity=float(quantity),
                order_type=normalized_type,  # type: ignore[arg-type]
                status="FILLED" if normalized_type == "market" else "OPEN",
                fill_price=fill_price if normalized_type == "market" else None,
                limit_price=float(price) if price is not None else None,
            )
            self.orders.append(order)

            if normalized_type == "market":
                self.cash -= quantity * fill_price
                self.cash -= commission
                self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity
                LOGGER.info(
                    "Paper Order: %s %.8f %s @ %.2f (simulated)",
                    "BUY" if quantity > 0 else "SELL",
                    abs(quantity),
                    symbol,
                    fill_price,
                )
            else:
                self.pending_orders[order.id] = order
            return order

        if self.client is None:
            raise LiveTradingNotConfiguredError("Live client not initialized.")

        side = Client.SIDE_BUY if quantity > 0 else Client.SIDE_SELL
        binance_type = Client.ORDER_TYPE_MARKET if normalized_type == "market" else Client.ORDER_TYPE_LIMIT
        payload = {
            "symbol": symbol,
            "side": side,
            "type": binance_type,
            "quantity": abs(quantity),
        }
        if normalized_type == "limit":
            payload["price"] = float(price) if price is not None else self._latest_price(symbol)
            payload["timeInForce"] = Client.TIME_IN_FORCE_GTC
        response = self._safe_live_call(self.client.create_order, **payload)
        order = Order(
            id=str(response.get("orderId")),
            symbol=symbol,
            quantity=float(quantity),
            order_type=normalized_type,  # type: ignore[arg-type]
            status="OPEN",
            fill_price=float(response.get("price") or 0.0) or None,
            limit_price=float(payload["price"]) if "price" in payload else None,
        )
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        if self.paper_mode:
            pending = self.pending_orders.pop(order_id, None)
            if pending is None:
                return False
            pending.status = "CANCELED"
            return True

        if self.client is None:
            return False
        order = next((item for item in self.orders if item.id == order_id), None)
        if order is None:
            return False
        self._safe_live_call(self.client.cancel_order, symbol=order.symbol, orderId=order.id)
        order.status = "CANCELED"
        return True

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_cash(self) -> float:
        return float(self.cash)

    def get_position_value_usd(self, symbol: str) -> float:
        qty = float(self.positions.get(symbol.upper(), 0.0))
        if qty == 0:
            return 0.0
        return qty * self._latest_price(symbol.upper())

    def get_account_value(self) -> float:
        return float(self.cash + sum(self.get_position_value_usd(symbol) for symbol in self.positions))

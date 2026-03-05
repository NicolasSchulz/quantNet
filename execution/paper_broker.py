from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from backtesting.cost_model import CostModel
from execution.base_broker import BaseBroker, Order


@dataclass
class PaperBroker(BaseBroker):
    """Paper trading simulator with next-bar market fills and limit logic."""

    initial_cash: float
    cost_model: CostModel
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    open_orders: dict[str, Order] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def place_order(
        self,
        symbol: str,
        quantity: float,
        order_type: str,
        price: float | None = None,
    ) -> Order:
        if quantity == 0:
            raise ValueError("quantity cannot be zero")
        if order_type not in {"market", "limit"}:
            raise ValueError("order_type must be 'market' or 'limit'")

        order = Order(
            id=str(uuid4()),
            symbol=symbol.upper(),
            quantity=float(quantity),
            order_type=order_type,
            status="OPEN",
            fill_price=None,
            limit_price=price,
        )
        self.open_orders[order.id] = order
        return order

    def cancel_order(self, order_id: str) -> None:
        order = self.open_orders.get(order_id)
        if order is None:
            raise KeyError(f"Order not found: {order_id}")
        if order.status != "OPEN":
            raise ValueError(f"Cannot cancel order with status {order.status}")
        order.status = "CANCELED"

    def process_bar(self, symbol: str, open_price: float, high_price: float, low_price: float) -> None:
        """Process one bar and fill eligible orders for symbol."""
        symbol = symbol.upper()
        self.last_prices[symbol] = open_price

        for order in list(self.open_orders.values()):
            if order.symbol != symbol or order.status != "OPEN":
                continue

            fill = None
            if order.order_type == "market":
                fill = open_price
            elif order.order_type == "limit" and order.limit_price is not None:
                if order.quantity > 0 and low_price <= order.limit_price:
                    fill = order.limit_price
                if order.quantity < 0 and high_price >= order.limit_price:
                    fill = order.limit_price

            if fill is None:
                continue

            direction = 1 if order.quantity > 0 else -1
            adjusted = self.cost_model.apply(fill, abs(order.quantity), direction)
            notional = abs(order.quantity) * adjusted
            commission = self.cost_model.calculate_commission(notional)

            self.cash -= order.quantity * adjusted
            self.cash -= commission
            self.positions[symbol] = self.positions.get(symbol, 0.0) + order.quantity

            order.fill_price = adjusted
            order.status = "FILLED"

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_cash(self) -> float:
        return float(self.cash)

    def get_account_value(self) -> float:
        value = self.cash
        for symbol, qty in self.positions.items():
            value += qty * self.last_prices.get(symbol, 0.0)
        return float(value)

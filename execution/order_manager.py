from __future__ import annotations

from dataclasses import dataclass, field

from execution.base_broker import BaseBroker, Order


@dataclass
class OrderManager:
    """Tracks order lifecycle and delegates execution to broker."""

    broker: BaseBroker
    orders: dict[str, Order] = field(default_factory=dict)

    def submit_order(
        self,
        symbol: str,
        quantity: float,
        order_type: str,
        price: float | None = None,
    ) -> Order:
        order = self.broker.place_order(symbol, quantity, order_type, price)
        order.status = "OPEN"
        self.orders[order.id] = order
        return order

    def cancel_order(self, order_id: str) -> None:
        self.broker.cancel_order(order_id)
        if order_id in self.orders:
            self.orders[order_id].status = "CANCELED"

    def get_open_orders(self) -> list[Order]:
        return [order for order in self.orders.values() if order.status == "OPEN"]

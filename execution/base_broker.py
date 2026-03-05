from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


OrderType = Literal["market", "limit"]
OrderStatus = Literal["NEW", "OPEN", "FILLED", "CANCELED", "REJECTED"]


@dataclass
class Order:
    id: str
    symbol: str
    quantity: float
    order_type: OrderType
    status: OrderStatus
    fill_price: float | None = None
    limit_price: float | None = None


class BaseBroker(ABC):
    """Abstract broker interface for paper/live adapters."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType,
        price: float | None = None,
    ) -> Order:
        """Submit a new order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Return current symbol -> quantity mapping."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return current available cash."""

    @abstractmethod
    def get_account_value(self) -> float:
        """Return current total account equity."""

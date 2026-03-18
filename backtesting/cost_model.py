from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Transaction cost model with commission, slippage, and spread."""

    asset_class: str = "equity"
    commission: float = 0.001
    slippage_bps: float = 5.0
    spread_bps: float = 2.0

    def apply(self, price: float, quantity: float, direction: int) -> float:
        """Return adjusted fill price including slippage and half-spread.

        Args:
            price: Mid/open price before costs.
            quantity: Order quantity (absolute not required).
            direction: +1 buy, -1 sell.
        """
        if price <= 0:
            raise ValueError("price must be positive")
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        _ = quantity

        total_bps = (self.slippage_bps + (self.spread_bps / 2.0)) / 10_000.0
        return price * (1.0 + direction * total_bps)

    def calculate_commission(self, notional: float) -> float:
        """Calculate proportional commission from trade notional."""
        if notional < 0:
            raise ValueError("notional must be non-negative")
        return notional * self.commission

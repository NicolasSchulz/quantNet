from __future__ import annotations

import pytest

from backtesting.cost_model import CostModel


def test_apply_buy_sell_adjustment() -> None:
    cm = CostModel(commission=0.001, slippage_bps=5, spread_bps=2)
    buy_price = cm.apply(price=100.0, quantity=10, direction=1)
    sell_price = cm.apply(price=100.0, quantity=10, direction=-1)
    assert buy_price > 100.0
    assert sell_price < 100.0


def test_commission_calculation() -> None:
    cm = CostModel(commission=0.001)
    assert cm.calculate_commission(10000.0) == pytest.approx(10.0)


def test_invalid_direction_raises() -> None:
    cm = CostModel()
    with pytest.raises(ValueError):
        cm.apply(price=100.0, quantity=1, direction=0)

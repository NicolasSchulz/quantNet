from __future__ import annotations


def check_max_position_size(position_value: float, portfolio_value: float, max_size: float) -> bool:
    if portfolio_value <= 0:
        return False
    return abs(position_value) / portfolio_value <= max_size


def check_concentration_limit(weights: dict[str, float], max_concentration: float) -> bool:
    return all(abs(weight) <= max_concentration for weight in weights.values())

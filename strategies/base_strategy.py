from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for strategy implementations."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return trading signals indexed like input data (+1, -1, 0)."""

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameter dictionary."""

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable strategy name."""

    def fit(self, data: pd.DataFrame) -> None:
        """Optional fit step for ML-based strategies."""
        _ = data
        return None

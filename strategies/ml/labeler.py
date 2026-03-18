from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class TripleBarrierLabeler:
    take_profit: float = 0.015
    stop_loss: float = 0.008
    max_holding: int = 48
    min_return: float = 0.001
    handle_imbalance: str = "weights"
    asset_class: str = "equity"

    def __post_init__(self) -> None:
        allowed = {"weights", "none", "undersample"}
        if self.handle_imbalance not in allowed:
            raise ValueError(f"handle_imbalance must be one of {sorted(allowed)}")
        if self.asset_class not in {"equity", "crypto"}:
            raise ValueError("asset_class must be 'equity' or 'crypto'")

    def label(self, data: pd.DataFrame) -> pd.Series:
        """Generate triple-barrier labels {-1, 0, +1}.

        Last max_holding rows are NaN because future path is unknown.
        """
        required = {"high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns for labeling: {sorted(missing)}")

        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        labels = pd.Series(np.nan, index=data.index, dtype="float")

        n = len(data)
        for i in range(0, n - self.max_holding):
            entry = close.iloc[i]
            tp_price = entry * (1.0 + self.take_profit)
            sl_price = entry * (1.0 - self.stop_loss)

            assigned = 0.0
            realized_ret = 0.0
            for j in range(i + 1, i + self.max_holding + 1):
                if high.iloc[j] >= tp_price:
                    assigned = 1.0
                    realized_ret = (tp_price / entry) - 1.0
                    break
                if low.iloc[j] <= sl_price:
                    assigned = -1.0
                    realized_ret = (sl_price / entry) - 1.0
                    break
                if j == i + self.max_holding:
                    assigned = 0.0
                    realized_ret = (close.iloc[j] / entry) - 1.0

            if abs(realized_ret) < self.min_return:
                assigned = 0.0

            labels.iloc[i] = assigned

        labels.attrs["take_profit"] = self.take_profit
        labels.attrs["stop_loss"] = self.stop_loss
        labels.attrs["max_holding"] = self.max_holding
        labels.attrs["min_return"] = self.min_return
        return labels

    def label_distribution(self, labels: pd.Series) -> dict[int | str, float]:
        valid = labels.dropna().astype(int)
        counts = valid.value_counts().to_dict()
        total = float(len(valid))
        out: dict[int | str, float] = {
            -1: float(counts.get(-1, 0)),
            0: float(counts.get(0, 0)),
            1: float(counts.get(1, 0)),
            "imbalance_ratio": 0.0,
        }
        if total > 0:
            shares = {k: out[k] / total for k in (-1, 0, 1)}
            out["imbalance_ratio"] = max(shares.values()) / max(min(shares.values()), 1e-12)
            for cls, share in shares.items():
                if share < 0.15:
                    LOGGER.warning(
                        "Class imbalance detected: class %s has %.2f%% of labels",
                        cls,
                        share * 100.0,
                    )
        return out

    def get_forward_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Column 'close' is required for forward returns")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["return_1d"] = close.shift(-1) / close - 1.0
        out["return_5d"] = close.shift(-5) / close - 1.0
        out["return_10d"] = close.shift(-10) / close - 1.0
        out["return_20d"] = close.shift(-20) / close - 1.0
        return out

    def get_class_weights(self, labels: pd.Series) -> dict[int, float] | None:
        if self.handle_imbalance != "weights":
            return None
        valid = labels.dropna().astype(int)
        if valid.empty:
            return None

        n_samples = float(len(valid))
        n_classes = 3.0
        counts = valid.value_counts().to_dict()
        weights: dict[int, float] = {}
        for cls in (-1, 0, 1):
            class_count = float(counts.get(cls, 0.0))
            if class_count <= 0:
                continue
            weights[cls] = n_samples / (n_classes * class_count)
        return weights

    def undersample_labels(self, labels: pd.Series, random_state: int = 42) -> pd.Series:
        if self.handle_imbalance != "undersample":
            return labels

        valid = labels.dropna().astype(int)
        if valid.empty:
            return labels

        try:
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:  # pragma: no cover - dependency/environment specific
            raise ImportError(
                "imbalanced-learn is required for handle_imbalance='undersample'."
            ) from exc

        X = pd.DataFrame({"idx": range(len(valid))}, index=valid.index)
        y = valid.values
        rus = RandomUnderSampler(random_state=random_state)
        _, y_res = rus.fit_resample(X, y)
        selected_indices = X.iloc[rus.sample_indices_].index

        out = pd.Series(np.nan, index=labels.index, dtype="float")
        out.loc[selected_indices] = y_res.astype(float)
        return out

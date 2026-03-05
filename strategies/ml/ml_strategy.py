from __future__ import annotations

import logging
from threading import Lock
from typing import Any
from pathlib import Path

import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.ml.model_registry import ModelRegistry
from strategies.ml.signal_filter import SignalFilter

LOGGER = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """Model-driven strategy implementing BaseStrategy.

    Signals at time T are computed from features that only use information
    available up to and including T (no look-ahead usage).
    """

    def __init__(
        self,
        symbol: str,
        model_registry: ModelRegistry,
        feature_engineer: Any,
        signal_filter: SignalFilter,
        model_id: str | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.model_registry = model_registry
        self.feature_engineer = feature_engineer
        self.signal_filter = signal_filter
        self._lock = Lock()

        payload = self.model_registry._read() if hasattr(self.model_registry, "_read") else {"models": {}, "default": {}}
        if model_id is None:
            model, feature_version = self.model_registry.get_default_model(self.symbol)
            self.model_id = payload.get("default", {}).get(self.symbol, "unknown")
        else:
            model = self.model_registry.get_model(model_id)
            self.model_id = model_id
            feature_version = str(payload.get("models", {}).get(model_id, {}).get("feature_version", "unknown"))

        self.model = model
        self.feature_version = feature_version
        self.model_info = payload.get("models", {}).get(self.model_id, {})
        self._last_single_metadata: dict[str, Any] = {}

        if getattr(self.feature_engineer, "scaler", None) is None:
            model_path = self.model_info.get("model_path")
            if model_path:
                scaler_path = Path(model_path).with_suffix(".scaler.joblib")
                if scaler_path.exists():
                    try:
                        self.feature_engineer.load_scaler(str(scaler_path))
                        LOGGER.info("Loaded feature scaler from %s", scaler_path)
                    except Exception as exc:  # pragma: no cover - IO/runtime dependent
                        LOGGER.warning("Failed to load scaler artifact %s: %s", scaler_path, exc)

        LOGGER.info(
            "Loaded ML model '%s' for %s with metrics=%s",
            self.model_id,
            self.symbol,
            self.model_info.get("metrics", {}),
        )

    def _prepare_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        # Engine currently passes close matrix for most strategies; for ML,
        # we expect full OHLCV. If close-only is provided, we create safe
        # placeholders to preserve interface compatibility.
        if {"open", "high", "low", "close", "volume"}.issubset(data.columns):
            ohlcv = data[["open", "high", "low", "close", "volume"]].copy()
        elif "close" in data.columns and len(data.columns) == 1:
            close = data["close"].astype(float)
            ohlcv = pd.DataFrame(
                {
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                },
                index=data.index,
            )
        elif self.symbol in data.columns:
            close = data[self.symbol].astype(float)
            ohlcv = pd.DataFrame(
                {
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                },
                index=data.index,
            )
        else:
            raise ValueError("MLStrategy requires OHLCV columns or a close series for target symbol")

        return ohlcv

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate {-1,0,+1} signals using model probabilities and filters."""
        with self._lock:
            ohlcv = self._prepare_ohlcv(data)
            features = self.feature_engineer.transform(ohlcv)
            probabilities = self.model.predict_proba(features)
            signals = self.signal_filter.filter(probabilities=probabilities, prices=ohlcv)
            return signals.astype(int)

    def generate_signal_single(self, data: pd.DataFrame) -> tuple[int, dict[str, Any]]:
        """Generate latest signal and metadata for live monitoring hooks."""
        with self._lock:
            ohlcv = self._prepare_ohlcv(data)
            features = self.feature_engineer.transform(ohlcv)
            probabilities = self.model.predict_proba(features)
            signals = self.signal_filter.filter(probabilities=probabilities, prices=ohlcv)

            ts = signals.index[-1]
            signal = int(signals.iloc[-1])
            proba_row = probabilities.loc[ts]

            regime = "unknown"
            if self.signal_filter.regime_filter is not None:
                regime_bool = self.signal_filter.regime_filter.is_bullish(ohlcv[["close"]].rename(columns={"close": self.signal_filter.regime_filter.benchmark.upper()}), ts)
                regime = "bullish" if regime_bool else "bearish"

            filtered_by = "none"
            if float(proba_row.max()) < self.signal_filter.min_confidence:
                filtered_by = "confidence"
            elif signal == 0 and float(proba_row["prob_long"]) >= self.signal_filter.min_confidence and regime == "bearish":
                filtered_by = "regime"

            metadata = {
                "prob_long": float(proba_row["prob_long"]),
                "prob_short": float(proba_row["prob_short"]),
                "prob_flat": float(proba_row["prob_flat"]),
                "confidence": float(proba_row.max()),
                "regime": regime,
                "filtered_by": filtered_by,
                "model_id": self.model_id,
            }
            self._last_single_metadata = metadata
            return signal, metadata

    def get_parameters(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "feature_version": self.feature_version,
            "min_confidence": self.signal_filter.min_confidence,
            "regime_filter": self.signal_filter.regime_filter is not None,
            "min_holding_days": self.signal_filter.min_holding_days,
            "model_metrics": self.model_info.get("metrics", {}),
        }

    def get_name(self) -> str:
        return f"MLStrategy_{self.symbol}_{self.model_id}"

    def fit(self, data: pd.DataFrame) -> None:
        _ = data
        LOGGER.warning(
            "MLStrategy.fit() hat keinen Effekt. Nutze model_training.py um das Modell neu zu trainieren."
        )

    def warmup_bars(self) -> int:
        return int(getattr(self.feature_engineer, "warmup_bars", 200))

    @property
    def expects_ohlcv(self) -> bool:
        return True

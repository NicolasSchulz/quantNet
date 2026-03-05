from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)


@dataclass
class TelegramAlerts:
    """Telegram alert helper with optional transport callback."""

    sender: Callable[[str], None] | None = None

    def _dispatch(self, message: str) -> None:
        if self.sender is not None:
            self.sender(message)
        else:
            LOGGER.info("Telegram message (dry-run): %s", message)

    def send_ml_signal_alert(self, symbol: str, signal: int, metadata: dict) -> None:
        direction_map = {1: "LONG (+1)", 0: "FLAT (0)", -1: "SHORT (-1)"}
        direction = direction_map.get(int(signal), "UNKNOWN")

        msg = (
            f"🤖 ML Signal: {symbol.upper()}\n"
            f"Direction: {direction}\n"
            f"Confidence: {float(metadata.get('confidence', 0.0)) * 100:.1f}%\n"
            f"Regime: {str(metadata.get('regime', 'unknown')).capitalize()}\n"
            f"Prob Long: {float(metadata.get('prob_long', 0.0)):.3f} | "
            f"Flat: {float(metadata.get('prob_flat', 0.0)):.3f} | "
            f"Short: {float(metadata.get('prob_short', 0.0)):.3f}\n"
            f"Model: {metadata.get('model_id', 'unknown')}"
        )
        self._dispatch(msg)

    def send_low_confidence_warning(self, symbol: str, max_prob: float) -> None:
        msg = (
            f"⚠️ Niedrige Modell-Konfidenz: {symbol.upper()} "
            f"({float(max_prob) * 100:.1f}%) – kein Trade"
        )
        self._dispatch(msg)

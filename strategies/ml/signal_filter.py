from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.metrics import sharpe_ratio
from strategies.filters.regime_filter import RegimeFilter

LOGGER = logging.getLogger(__name__)


@dataclass
class SignalFilter:
    """Turns raw model probabilities into tradable signals."""

    min_confidence: float = 0.45
    regime_filter: RegimeFilter | None = None
    min_holding_days: int = 3
    max_positions_per_day: int = 1
    signal_smoothing: bool = True

    def __post_init__(self) -> None:
        self._last_stats: dict[str, float | int] = {
            "pct_long": 0.0,
            "pct_short": 0.0,
            "pct_flat": 0.0,
            "avg_holding_days": 0.0,
            "n_position_changes": 0,
            "signals_filtered_by_confidence": 0,
            "signals_filtered_by_regime": 0,
        }

    def _confidence_filter(self, probabilities: pd.DataFrame) -> tuple[pd.Series, int]:
        max_prob = probabilities.max(axis=1)
        argmax = probabilities.idxmax(axis=1)
        mapping = {"prob_short": -1, "prob_flat": 0, "prob_long": 1}
        signal = argmax.map(mapping).astype(int)

        low_conf = max_prob < self.min_confidence
        signal.loc[low_conf] = 0
        return signal, int(low_conf.sum())

    def _apply_regime(self, signal: pd.Series, prices: pd.DataFrame) -> tuple[pd.Series, int]:
        if self.regime_filter is None:
            return signal, 0

        benchmark = self.regime_filter.benchmark.upper()
        if benchmark in prices.columns:
            regime_input = prices[[benchmark]].copy()
        elif "close" in prices.columns:
            # Accept OHLCV frames by mapping close to benchmark alias.
            regime_input = prices[["close"]].rename(columns={"close": benchmark})
        else:
            raise ValueError(
                f"Regime filter requires either benchmark column '{benchmark}' "
                "or OHLCV column 'close'."
            )

        regime = self.regime_filter.get_regime_series(regime_input)
        aligned = regime.reindex(signal.index).fillna(False)

        filtered = signal.copy()
        blocked = (filtered == 1) & (~aligned)
        filtered.loc[blocked] = 0
        return filtered, int(blocked.sum())

    def _apply_min_holding(self, signal: pd.Series) -> tuple[pd.Series, dict[pd.Timestamp, str]]:
        if self.min_holding_days <= 1 or signal.empty:
            return signal, {}

        out = signal.copy()
        current = int(out.iloc[0])
        hold_count = 1
        reasons: dict[pd.Timestamp, str] = {}

        for i in range(1, len(out)):
            proposed = int(out.iloc[i])
            if proposed == current:
                hold_count += 1
                continue

            if hold_count < self.min_holding_days:
                out.iloc[i] = current
                hold_count += 1
                reasons[out.index[i]] = "holding"
            else:
                current = proposed
                hold_count = 1

        return out, reasons

    def _apply_smoothing(self, signal: pd.Series) -> pd.Series:
        if len(signal) < 3:
            return signal

        out = signal.copy()
        for i in range(2, len(out)):
            window = out.iloc[i - 2 : i + 1]
            mode = int(window.mode().iloc[0]) if not window.mode().empty else int(out.iloc[i])
            if int(out.iloc[i]) != mode:
                out.iloc[i] = int(out.iloc[i - 1])
        return out

    def filter(
        self,
        probabilities: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.Series:
        """Apply confidence/regime/holding/smoothing filters to model probabilities."""
        required_cols = ["prob_short", "prob_flat", "prob_long"]
        if list(probabilities.columns) != required_cols:
            raise ValueError(f"probabilities columns must be exactly {required_cols}")

        base, conf_filtered = self._confidence_filter(probabilities)
        regime_applied, regime_filtered = self._apply_regime(base, prices)
        held, _reasons = self._apply_min_holding(regime_applied)
        smoothed = self._apply_smoothing(held) if self.signal_smoothing else held

        # max_positions_per_day reserved for multi-asset extension; keep interface stable.
        _ = self.max_positions_per_day

        self._last_stats = self._compute_stats(smoothed, conf_filtered, regime_filtered)
        return smoothed.astype(int)

    def _compute_stats(self, signals: pd.Series, conf_count: int, regime_count: int) -> dict[str, float | int]:
        pct_long = float((signals == 1).mean()) if len(signals) else 0.0
        pct_short = float((signals == -1).mean()) if len(signals) else 0.0
        pct_flat = float((signals == 0).mean()) if len(signals) else 0.0

        changes = signals.ne(signals.shift(1)).sum() - (1 if len(signals) else 0)
        changes = int(max(0, changes))

        run_lengths: list[int] = []
        if len(signals):
            run = 1
            for i in range(1, len(signals)):
                if int(signals.iloc[i]) == int(signals.iloc[i - 1]):
                    run += 1
                else:
                    run_lengths.append(run)
                    run = 1
            run_lengths.append(run)

        avg_holding = float(np.mean(run_lengths)) if run_lengths else 0.0

        return {
            "pct_long": pct_long,
            "pct_short": pct_short,
            "pct_flat": pct_flat,
            "avg_holding_days": avg_holding,
            "n_position_changes": changes,
            "signals_filtered_by_confidence": int(conf_count),
            "signals_filtered_by_regime": int(regime_count),
        }

    def get_signal_stats(self, signals: pd.Series) -> dict[str, float | int]:
        """Return signal mix and churn stats; falls back to last computed filter stats."""
        if signals is None or signals.empty:
            return dict(self._last_stats)
        base = self._compute_stats(signals.astype(int), 0, 0)
        base["signals_filtered_by_confidence"] = int(self._last_stats["signals_filtered_by_confidence"])
        base["signals_filtered_by_regime"] = int(self._last_stats["signals_filtered_by_regime"])
        return base

    def tune_confidence_threshold(
        self,
        probabilities: pd.DataFrame,
        returns: pd.Series,
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        if thresholds is None:
            thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

        rows: list[dict[str, float]] = []
        original = self.min_confidence

        dummy_prices = pd.DataFrame(index=probabilities.index)
        if self.regime_filter is not None:
            # Fallback to bullish for tuning when benchmark prices are unavailable.
            dummy_prices[self.regime_filter.benchmark.upper()] = 1.0

        for thr in thresholds:
            self.min_confidence = float(thr)
            sig = self.filter(probabilities=probabilities, prices=dummy_prices)
            ret = returns.reindex(sig.index).fillna(0.0)
            pnl = sig.shift(1).fillna(0.0) * ret
            trades = sig.diff().abs().fillna(0.0)

            rows.append(
                {
                    "threshold": float(thr),
                    "sharpe": float(sharpe_ratio(pnl, risk_free_rate=0.0)),
                    "n_trades": float((trades > 0).sum()),
                    "pct_flat": float((sig == 0).mean()),
                }
            )

        self.min_confidence = original
        out = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
        return out

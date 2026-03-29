from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler

try:
    import pandas_ta as ta
except ImportError as exc:  # pragma: no cover - dependency/environment specific
    raise ImportError(
        "pandas-ta is required for FeatureEngineer. Install dependency 'pandas-ta'."
    ) from exc

LOGGER = logging.getLogger(__name__)

TREND_FEATURES = {
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",
    "sma20_vs_sma50",
    "sma50_vs_sma200",
}
MOMENTUM_FEATURES = {
    "rsi_14",
    "rsi_28",
    "macd",
    "macd_signal",
    "macd_hist",
    "roc_5",
    "roc_10",
    "roc_20",
    "roc_60",
    "stoch_k",
    "stoch_d",
}
VOLATILITY_FEATURES = {
    "atr_14",
    "atr_28",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_position",
    "realized_vol_10",
    "realized_vol_20",
    "vol_ratio",
}
VOLUME_FEATURES = {
    "volume_sma20",
    "volume_sma5",
    "obv_normalized",
    "vwap_distance",
}
CANDLE_FEATURES = {
    "body_size",
    "upper_wick",
    "lower_wick",
    "gap",
}
CRYPTO_FEATURES = {
    "funding_rate_proxy",
    "volume_usd_ratio",
    "price_vs_ath_proxy",
    "volatility_regime",
    "close_rank_252d",
    "days_since_rolling_high_252",
    "sma_100_vs_sma200",
    "trend_slope_200",
}
STRUCTURE_FEATURES = {
    "mss_candidate",
    "mss_direction",
    "market_structure_bias",
    "choch_bullish_flag",
    "choch_bearish_flag",
}

GROUP_TO_FEATURES = {
    "trend": TREND_FEATURES,
    "momentum": MOMENTUM_FEATURES,
    "volatility": VOLATILITY_FEATURES,
    "volume": VOLUME_FEATURES,
    "candle": CANDLE_FEATURES,
    "crypto": CRYPTO_FEATURES,
    "structure": STRUCTURE_FEATURES,
}

# Keep naturally bounded indicators unchanged.
BOUNDED_FEATURES = {
    "rsi_14",
    "rsi_28",
    "stoch_k",
    "stoch_d",
    "bb_position",
    "body_size",
    "upper_wick",
    "lower_wick",
}


class FeatureEngineer:
    """Compute model features from normalized OHLCV bars.

    The input should contain at least 250 bars because the longest feature
    lookback is 200 bars and additional bars improve stability.
    """

    def __init__(self, config: dict[str, Any] | None = None, asset_class: str = "equity") -> None:
        if config is None:
            with Path("config/settings.yaml").open("r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)
            ml_cfg = settings.get("ml", {}).get("features", {})
            lb = ml_cfg.get("lookback_periods", {})
            groups_key = "crypto_groups" if asset_class == "crypto" else "groups"
            config = {
                "feature_groups": ml_cfg.get(groups_key, ml_cfg.get("groups", ["trend", "momentum", "volatility", "volume", "candle"])),
                "normalization": ml_cfg.get("normalization", "robust"),
                "lookback_periods": lb,
                "warmup_bars": int(lb.get("warmup_bars", ml_cfg.get("warmup_bars", 200))),
            }

        self.config = config
        self.asset_class = str(asset_class).lower()
        self.feature_groups: list[str] = list(config.get("feature_groups", config.get("groups", [])))
        if self.asset_class == "crypto" and "crypto" not in self.feature_groups:
            self.feature_groups.append("crypto")
        self.normalization: str = str(config.get("normalization", "robust")).lower()
        self.warmup_bars: int = int(config.get("warmup_bars", 200))
        self.lookback_periods: dict[str, int] = dict(config.get("lookback_periods", {}))
        self.mss_config: dict[str, Any] = dict(config.get("mss_strategy", {}))
        self.scaler: RobustScaler | None = None
        self._scale_columns: list[str] = []
        self.feature_names_: list[str] = []

    def _lb(self, key: str, default: int) -> int:
        return int(self.lookback_periods.get(key, default))

    def _validate_input(self, data: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"FeatureEngineer requires OHLCV columns. Missing: {sorted(missing)}")
        if len(data) < 250:
            raise ValueError("FeatureEngineer requires at least 250 rows of input data.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Input index must be a pandas DatetimeIndex.")

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)
        df = data.copy().sort_index()

        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float)

        out = pd.DataFrame(index=df.index)

        # Trend
        sma_short = self._lb("sma_short", 20)
        sma_medium = self._lb("sma_medium", 50)
        sma_long = self._lb("sma_long", 200)
        macd_fast = self._lb("macd_fast", 12)
        macd_slow = self._lb("macd_slow", 26)
        macd_signal = self._lb("macd_signal", 9)
        rsi_len = self._lb("rsi", 14)
        atr_len = self._lb("atr", 14)
        bb_len = self._lb("bb", 20)
        rv_short = self._lb("realized_vol_short", 10)
        rv_long = self._lb("realized_vol_long", 20)

        out["sma_20"] = ta.sma(c, length=sma_short)
        out["sma_50"] = ta.sma(c, length=sma_medium)
        out["sma_200"] = ta.sma(c, length=sma_long)
        out["ema_12"] = ta.ema(c, length=macd_fast)
        out["ema_26"] = ta.ema(c, length=macd_slow)
        out["price_vs_sma20"] = (c / out["sma_20"]) - 1.0
        out["price_vs_sma50"] = (c / out["sma_50"]) - 1.0
        out["price_vs_sma200"] = (c / out["sma_200"]) - 1.0
        out["sma20_vs_sma50"] = (out["sma_20"] / out["sma_50"]) - 1.0
        out["sma50_vs_sma200"] = (out["sma_50"] / out["sma_200"]) - 1.0

        # Momentum
        out["rsi_14"] = ta.rsi(c, length=rsi_len)
        out["rsi_28"] = ta.rsi(c, length=max(rsi_len * 2, 2))
        macd_df = ta.macd(c, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        out["macd"] = macd_df.iloc[:, 0] / c
        out["macd_hist"] = macd_df.iloc[:, 1] / c
        out["macd_signal"] = macd_df.iloc[:, 2] / c
        out["roc_5"] = ta.roc(c, length=5)
        out["roc_10"] = ta.roc(c, length=10)
        out["roc_20"] = ta.roc(c, length=20)
        out["roc_60"] = ta.roc(c, length=60)
        stoch_df = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
        out["stoch_k"] = stoch_df.iloc[:, 0]
        out["stoch_d"] = stoch_df.iloc[:, 1]

        # Volatility
        out["atr_14"] = ta.atr(h, l, c, length=atr_len) / c
        out["atr_28"] = ta.atr(h, l, c, length=max(atr_len * 2, 2)) / c
        bb = ta.bbands(c, length=bb_len, std=2)
        bb_lower = bb.iloc[:, 0]
        bb_mid = bb.iloc[:, 1]
        bb_upper = bb.iloc[:, 2]
        bb_std = (bb_upper - bb_lower) / 4.0
        out["bb_upper"] = (bb_upper - bb_mid) / (bb_std + 1e-10)
        out["bb_lower"] = (bb_lower - bb_mid) / (bb_std + 1e-10)
        out["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        out["bb_position"] = (c - bb_lower) / ((bb_upper - bb_lower) + 1e-10)
        log_returns = np.log(c / c.shift(1))
        out["realized_vol_10"] = log_returns.rolling(rv_short).std(ddof=0) * np.sqrt(252.0 * 6.5)
        out["realized_vol_20"] = log_returns.rolling(rv_long).std(ddof=0) * np.sqrt(252.0 * 6.5)
        out["vol_ratio"] = out["realized_vol_10"] / (out["realized_vol_20"] + 1e-10)

        # Volume
        vol_sma20 = ta.sma(v, length=20)
        vol_sma5 = ta.sma(v, length=5)
        out["volume_sma20"] = v / (vol_sma20 + 1e-10)
        out["volume_sma5"] = v / (vol_sma5 + 1e-10)
        obv = ta.obv(c, v)
        obv_sma20 = ta.sma(obv, length=20)
        out["obv_normalized"] = (obv / (obv_sma20 + 1e-10)) - 1.0
        vwma_20 = ta.vwma(c, v, length=20)
        out["vwap_distance"] = (c - vwma_20) / (c + 1e-10)

        # Candle
        hl_range = (h - l) + 1e-10
        out["body_size"] = (c - o).abs() / hl_range
        out["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / hl_range
        out["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / hl_range
        out["gap"] = (o - c.shift(1)) / (c.shift(1) + 1e-10)

        if self.asset_class == "crypto":
            taker_volume = pd.to_numeric(df.get("taker_volume", pd.Series(np.nan, index=df.index)), errors="coerce")
            volume_usd = pd.to_numeric(df.get("volume_usd", v * c), errors="coerce")
            realized_vol_24h = log_returns.rolling(24).std(ddof=0) * np.sqrt(365.0 * 24.0)
            realized_vol_7d = log_returns.rolling(24 * 7).std(ddof=0) * np.sqrt(365.0 * 24.0)
            rolling_year = 24 * 252

            def _close_rank(values: np.ndarray) -> float:
                if len(values) == 0:
                    return np.nan
                s = pd.Series(values)
                return float(s.rank(pct=True).iloc[-1])

            def _bars_since_high(values: np.ndarray) -> float:
                if len(values) == 0:
                    return np.nan
                return float((len(values) - 1) - int(np.argmax(values)))

            # Older cached crypto bars may not include taker_volume yet.
            # Use a neutral value instead of dropping the whole feature matrix.
            funding_rate_proxy = (taker_volume / (v + 1e-10)) - 0.5
            out["funding_rate_proxy"] = funding_rate_proxy.fillna(0.0)
            out["volume_usd_ratio"] = volume_usd / (ta.sma(volume_usd, length=20) + 1e-10)
            out["price_vs_ath_proxy"] = c / (c.rolling(365).max() + 1e-10)
            out["volatility_regime"] = realized_vol_24h / (realized_vol_7d + 1e-10)
            sma_100 = ta.sma(c, length=100)
            out["close_rank_252d"] = c.rolling(rolling_year).apply(_close_rank, raw=True)
            out["days_since_rolling_high_252"] = c.rolling(rolling_year).apply(_bars_since_high, raw=True) / float(rolling_year)
            out["sma_100_vs_sma200"] = (sma_100 / (out["sma_200"] + 1e-10)) - 1.0
            out["trend_slope_200"] = ta.slope(out["sma_200"], length=20) / (out["sma_200"] + 1e-10)

        if "structure" in self.feature_groups:
            from strategies.ml.mss_entry_strategy import MSSEntryStrategy

            mss = MSSEntryStrategy(
                swing_n=int(self.mss_config.get("swing_n", 4)),
                atr_period=int(self.mss_config.get("atr_period", 14)),
                tp_mult=float(self.mss_config.get("tp_mult", 3.0)),
                sl_mult=float(self.mss_config.get("sl_mult", 1.5)),
                max_hold=int(self.mss_config.get("max_hold", 48)),
                pullback_timeout=int(self.mss_config.get("pullback_timeout", 5)),
                choch_min_break_atr=float(self.mss_config.get("choch_min_break_atr", 0.15)),
                choch_min_body_fraction=float(self.mss_config.get("choch_min_body_fraction", 0.35)),
                pullback_retest_atr=float(self.mss_config.get("pullback_retest_atr", 0.25)),
                pullback_max_overshoot_atr=float(self.mss_config.get("pullback_max_overshoot_atr", 0.35)),
                confirmation_break_atr=float(self.mss_config.get("confirmation_break_atr", 0.05)),
            )
            structure = mss.apply_mss_filter(df)
            out["mss_candidate"] = structure["mss_entry_candidate"].astype(float)
            out["mss_direction"] = structure["entry_direction_bias"].astype(float)
            out["market_structure_bias"] = (
                structure["market_structure"]
                .map({"downtrend": -1.0, "undefined": 0.0, "uptrend": 1.0})
                .fillna(0.0)
                .astype(float)
            )
            out["choch_bullish_flag"] = structure["choch_bullish"].astype(float)
            out["choch_bearish_flag"] = structure["choch_bearish"].astype(float)

        selected_features = sorted(
            set().union(*(GROUP_TO_FEATURES[g] for g in self.feature_groups if g in GROUP_TO_FEATURES))
        )
        out = out[selected_features]

        # Warmup removal: longest lookback is 200 bars.
        out = out.iloc[self.warmup_bars :].dropna(how="any")
        out = out.astype("float64")
        self.feature_names_ = list(out.columns)
        return out

    def fit_scaler(self, X: pd.DataFrame) -> RobustScaler:
        if self.normalization != "robust":
            scaler = RobustScaler(with_centering=False, with_scaling=False)
            scaler.fit(np.zeros((1, len(X.columns) or 1)))
            return scaler

        scale_cols = [c for c in X.columns if c not in BOUNDED_FEATURES]
        scaler = RobustScaler()
        if scale_cols:
            scaler.fit(X[scale_cols])
        self._scale_columns = scale_cols
        return scaler

    def scale_features(
        self,
        X: pd.DataFrame,
        scaler: RobustScaler,
    ) -> pd.DataFrame:
        if self.normalization != "robust":
            return X.astype("float64")

        scale_cols = [c for c in X.columns if c not in BOUNDED_FEATURES]
        if not scale_cols:
            return X.astype("float64")

        result = X.copy()
        result[scale_cols] = scaler.transform(result[scale_cols])
        result = result.astype("float64")
        return result

    def _warn_deprecated_transform(self) -> None:
        msg = (
            "transform() / fit_transform() ist deprecated. Nutze compute_features() + "
            "fit_scaler() + scale_features() fuer korrektes Walk-Forward Setup."
        )
        LOGGER.warning(msg)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    def _scaled_features(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if self.normalization != "robust":
            return features.astype("float64")

        if fit:
            self.scaler = self.fit_scaler(features)
            scale_cols = [c for c in features.columns if c not in BOUNDED_FEATURES]
            LOGGER.info("Fitted RobustScaler on %d features", len(scale_cols))
            return self.scale_features(features, self.scaler)

        if self.scaler is None:
            LOGGER.warning("No fitted scaler found. Returning unscaled features for transform().")
            return features.astype("float64")

        return self.scale_features(features, self.scaler)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform normalized OHLCV into engineered features.

        Returns a feature-only DataFrame indexed by timestamp.
        """
        self._warn_deprecated_transform()
        features = self.compute_features(data)
        return self._scaled_features(features, fit=False)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on this dataset and return transformed features."""
        self._warn_deprecated_transform()
        features = self.compute_features(data)
        return self._scaled_features(features, fit=True)

    def save_scaler(self, path: str) -> None:
        if self.scaler is None:
            raise ValueError("No scaler available to save. Call fit_transform() first.")
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        self.scaler = joblib.load(path)

    def get_feature_names(self) -> list[str]:
        return list(self.feature_names_)

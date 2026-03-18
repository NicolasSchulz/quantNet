from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from data.features.feature_store import FeatureStore
from data.ingestion.feed_factory import FeedFactory
from data.normalizer import normalize_ohlcv
from data.storage.parquet_store import ParquetStore
from strategies.ml.feature_engineer import FeatureEngineer
from strategies.ml.labeler import TripleBarrierLabeler


def load_settings(path: str = "config/settings.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_ohlcv(
    symbol: str,
    asset_class: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    feed,
    store: ParquetStore,
) -> pd.DataFrame:
    start_str = pd.Timestamp(start).date().isoformat()
    end_str = pd.Timestamp(end).date().isoformat()
    if store.exists(symbol, interval):
        cached = store.load(symbol=symbol, interval=interval, start=start, end=end)
        if not cached.empty:
            return cached

    raw = feed.fetch_historical(symbol=symbol, start=start_str, end=end_str, interval=interval)
    normalized = normalize_ohlcv(raw, symbol=symbol, asset_class=asset_class, interval=interval)
    store.save(normalized, symbol=symbol, interval=interval)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature + label dataset for one symbol")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--asset-class", default="equity", choices=["equity", "crypto"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("feature_pipeline")

    settings = load_settings()
    interval = settings["data"]["intervals"]["primary"]
    ml_features_cfg = settings["ml"]["features"]
    ml_label_cfg = settings["ml"]["labeling"]

    symbol = args.symbol.upper()
    asset_class = args.asset_class.lower()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    feed, bar_store = FeedFactory.create_with_cache(config=settings, symbol=symbol)

    ohlcv = _load_ohlcv(
        symbol=symbol,
        asset_class=asset_class,
        start=start,
        end=end,
        interval=interval,
        feed=feed,
        store=bar_store,
    )

    engineer = FeatureEngineer(
        config={
            "feature_groups": ml_features_cfg["crypto_groups"] if asset_class == "crypto" else ml_features_cfg["groups"],
            "normalization": ml_features_cfg["normalization"],
            "lookback_periods": ml_features_cfg.get("lookback_periods", {}),
            "warmup_bars": int(ml_features_cfg.get("lookback_periods", {}).get("warmup_bars", 200)),
        },
        asset_class=asset_class,
    )
    features = engineer.compute_features(ohlcv)

    label_cfg = ml_label_cfg[asset_class]
    labeler = TripleBarrierLabeler(
        take_profit=float(label_cfg["take_profit"]),
        stop_loss=float(label_cfg["stop_loss"]),
        max_holding=int(label_cfg["max_holding"]),
        min_return=float(label_cfg["min_return"]),
        handle_imbalance=str(label_cfg.get("handle_imbalance", "weights")),
        asset_class=asset_class,
    )
    labels = labeler.label(ohlcv)

    aligned_labels = labels.reindex(features.index)
    distribution = labeler.label_distribution(aligned_labels)

    feature_store = FeatureStore(base_path=settings["data"]["storage_path"] + "/features")
    version = str(ml_features_cfg["feature_version"])
    feature_store.save(raw_features=features, labels=aligned_labels, symbol=symbol, feature_version=version)

    out_path = Path(feature_store.base_path) / symbol / version
    valid = aligned_labels.dropna().astype(int)
    total = len(valid)

    logger.info("Label distribution: %s", distribution)
    logger.info("Features gespeichert: UNSKALIERT (korrekt fuer Walk-Forward)")

    print(f"Symbol:             {symbol}")
    print(f"Asset-Klasse:       {asset_class}")
    print(f"Zeitraum:           {args.start} bis {args.end}")
    print(f"Bars nach Warmup:   {len(features)}")
    print(f"Features:           {features.shape[1]}")
    print(f"Label Long  (+1):   {(valid == 1).sum()} ({(((valid == 1).sum() / total) * 100 if total else 0):.1f}%)")
    print(f"Label Short (-1):   {(valid == -1).sum()} ({(((valid == -1).sum() / total) * 100 if total else 0):.1f}%)")
    print(f"Label Flat  ( 0):   {(valid == 0).sum()} ({(((valid == 0).sum() / total) * 100 if total else 0):.1f}%)")
    print(f"Gespeichert unter:  {out_path}/")
    if asset_class == "equity":
        print("Hinweis: Polygon Starter Plan: max. 2 Jahre Historie verfügbar.")
        print("Für längere Historie: Polygon Developer Plan (79$/Monat).")


if __name__ == "__main__":
    main()

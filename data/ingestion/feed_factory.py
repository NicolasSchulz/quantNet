from __future__ import annotations

import logging
import os
from typing import Any

import yaml
from dotenv import load_dotenv

from data.ingestion.base_feed import BaseFeed
from data.ingestion.yahoo_feed import YahooFeed
from data.storage.parquet_store import ParquetStore

LOGGER = logging.getLogger(__name__)


class MissingApiKeyError(RuntimeError):
    pass


class FeedFactory:
    @staticmethod
    def _load_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
        if config is not None:
            return config
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def create(source: str | None = None, config: dict[str, Any] | None = None) -> BaseFeed:
        settings = FeedFactory._load_settings(config)
        data_cfg = settings.get("data", {})
        feed_cfg = data_cfg.get("feed", {})
        requested_source = source
        source = source or feed_cfg.get("primary", "auto")
        fallback_source = feed_cfg.get("fallback", "yahoo")
        allow_primary_fallback = requested_source is None and source != "auto"

        load_dotenv("config/secrets.env")

        if source == "polygon":
            try:
                from data.ingestion.polygon_feed import PolygonAuthError, PolygonFeed
            except ImportError as exc:
                if allow_primary_fallback and fallback_source == "yahoo":
                    LOGGER.warning(
                        "Polygon dependency missing (%s). Fallback auf Yahoo Finance.",
                        exc,
                    )
                    return YahooFeed()
                raise RuntimeError(
                    "polygon-api-client not installed. Add it to requirements and install dependencies."
                ) from exc
            if not os.getenv("POLYGON_API_KEY"):
                if allow_primary_fallback and fallback_source == "yahoo":
                    LOGGER.warning(
                        "POLYGON_API_KEY missing. Fallback auf Yahoo Finance."
                    )
                    return YahooFeed()
                raise MissingApiKeyError(
                    "POLYGON_API_KEY nicht in secrets.env gefunden. "
                    "Kopiere config/secrets.env.example nach config/secrets.env und trage deinen Key ein."
                )
            poly_cfg = data_cfg.get("polygon", {})
            try:
                return PolygonFeed(
                    api_key=os.getenv("POLYGON_API_KEY"),
                    rate_limit_pause=float(feed_cfg.get("rate_limit_pause", 12.5)),
                    max_retries=int(poly_cfg.get("max_retries", 3)),
                    adjusted=bool(poly_cfg.get("adjusted", True)),
                )
            except Exception as exc:
                if allow_primary_fallback and fallback_source == "yahoo":
                    LOGGER.warning("Polygon unavailable (%s). Fallback auf Yahoo Finance.", exc)
                    return YahooFeed()
                raise

        if source == "yahoo":
            return YahooFeed()

        if source == "auto":
            try:
                from data.ingestion.polygon_feed import PolygonAuthError, PolygonFeed
                feed = FeedFactory.create("polygon", settings)
                if isinstance(feed, PolygonFeed) and feed.validate_api_key():
                    return feed
                LOGGER.warning("Polygon API key validation failed, fallback auf Yahoo Finance")
            except (MissingApiKeyError, PolygonAuthError, Exception) as exc:
                LOGGER.warning("Polygon nicht verfügbar (%s), fallback auf Yahoo Finance", exc)
            return YahooFeed()

        raise ValueError("source must be one of: polygon, yahoo, auto")

    @staticmethod
    def create_with_cache(
        source: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[BaseFeed, ParquetStore]:
        settings = FeedFactory._load_settings(config)
        feed = FeedFactory.create(source=source, config=settings)
        store = ParquetStore(storage_path=settings["data"]["storage_path"])
        return feed, store

from __future__ import annotations

from data.universes.etf_universe import EtfUniverse


def test_universe_has_30_symbols() -> None:
    universe = EtfUniverse()
    symbols = universe.get_symbols()
    assert len(symbols) == 30
    assert "XLK" in symbols
    assert "HYG" in symbols


def test_universe_metadata_schema() -> None:
    metadata = EtfUniverse().get_metadata()
    assert list(metadata.columns) == ["symbol", "name", "category", "asset_class"]
    assert (metadata["asset_class"] == "etf").all()


def test_filter_by_category() -> None:
    sectors = EtfUniverse().filter_by_category("sector")
    assert len(sectors) == 11
    assert "XLV" in sectors

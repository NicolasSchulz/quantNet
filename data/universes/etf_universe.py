from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EtfUniverse:
    """Curated list of liquid US sector, factor, and macro ETFs."""

    _rows: tuple[tuple[str, str, str, str], ...] = (
        ("XLK", "Technology Select Sector SPDR Fund", "sector", "etf"),
        ("XLV", "Health Care Select Sector SPDR Fund", "sector", "etf"),
        ("XLF", "Financial Select Sector SPDR Fund", "sector", "etf"),
        ("XLE", "Energy Select Sector SPDR Fund", "sector", "etf"),
        ("XLI", "Industrial Select Sector SPDR Fund", "sector", "etf"),
        ("XLY", "Consumer Discretionary Select Sector SPDR Fund", "sector", "etf"),
        ("XLP", "Consumer Staples Select Sector SPDR Fund", "sector", "etf"),
        ("XLU", "Utilities Select Sector SPDR Fund", "sector", "etf"),
        ("XLB", "Materials Select Sector SPDR Fund", "sector", "etf"),
        ("XLRE", "Real Estate Select Sector SPDR Fund", "sector", "etf"),
        ("XLC", "Communication Services Select Sector SPDR Fund", "sector", "etf"),
        ("VTV", "Vanguard Value ETF", "factor_theme", "etf"),
        ("VUG", "Vanguard Growth ETF", "factor_theme", "etf"),
        ("IJR", "iShares Core S&P Small-Cap ETF", "factor_theme", "etf"),
        ("IWM", "iShares Russell 2000 ETF", "factor_theme", "etf"),
        ("EFA", "iShares MSCI EAFE ETF", "factor_theme", "etf"),
        ("EEM", "iShares MSCI Emerging Markets ETF", "factor_theme", "etf"),
        ("GLD", "SPDR Gold Shares", "factor_theme", "etf"),
        ("TLT", "iShares 20+ Year Treasury Bond ETF", "factor_theme", "etf"),
        ("DBC", "Invesco DB Commodity Index Tracking Fund", "factor_theme", "etf"),
        ("VNQ", "Vanguard Real Estate ETF", "factor_theme", "etf"),
        ("EWJ", "iShares MSCI Japan ETF", "international_macro", "etf"),
        ("EWZ", "iShares MSCI Brazil ETF", "international_macro", "etf"),
        ("EWC", "iShares MSCI Canada ETF", "international_macro", "etf"),
        ("EWA", "iShares MSCI Australia ETF", "international_macro", "etf"),
        ("FXI", "iShares China Large-Cap ETF", "international_macro", "etf"),
        ("INDA", "iShares MSCI India ETF", "international_macro", "etf"),
        ("VGK", "Vanguard FTSE Europe ETF", "international_macro", "etf"),
        ("AGG", "iShares Core U.S. Aggregate Bond ETF", "international_macro", "etf"),
        ("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "international_macro", "etf"),
    )

    def get_symbols(self) -> list[str]:
        return [row[0] for row in self._rows]

    def get_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=["symbol", "name", "category", "asset_class"])

    def filter_by_category(self, category: str) -> list[str]:
        category_lower = category.strip().lower()
        metadata = self.get_metadata()
        filtered = metadata[metadata["category"].str.lower() == category_lower]
        return filtered["symbol"].tolist()

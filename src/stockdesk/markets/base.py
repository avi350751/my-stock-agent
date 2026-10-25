"""MarketAdapter: the single seam that keeps the agents market-agnostic.

Adding a market (e.g. US) means implementing this interface in a new module and
registering it in markets/registry.py. Agents and orchestration never import a
concrete market.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time

import pandas as pd


@dataclass(frozen=True)
class ResolvedSymbol:
    symbol: str          # exchange symbol (RELIANCE)
    ticker: str          # data-provider ticker (RELIANCE.NS)
    name: str            # company name
    industry: str = ""


@dataclass
class NewsItem:
    title: str
    link: str = ""
    source: str = ""
    published: datetime | None = None
    summary: str = ""


class AmbiguousSymbol(Exception):
    def __init__(self, query: str, candidates: list[ResolvedSymbol]):
        self.query = query
        self.candidates = candidates
        names = ", ".join(f"{c.symbol} ({c.name})" for c in candidates[:6])
        super().__init__(f"'{query}' matches several stocks: {names}. Please pass the exact symbol.")


class SymbolNotFound(Exception):
    pass


@dataclass(frozen=True)
class MarketInfo:
    key: str                  # "NSE"
    display_name: str         # "National Stock Exchange of India"
    currency: str             # "INR"
    currency_symbol: str      # "₹"
    timezone: str             # "Asia/Kolkata"
    close_time: time          # local close
    index_ticker: str         # "^NSEI"
    index_name: str           # "NIFTY 50"
    value_unit: str = "cr"    # traded-value display unit
    value_divisor: float = 1e7
    universes: tuple[str, ...] = field(default_factory=tuple)
    global_benchmarks: tuple[tuple[str, str], ...] = field(default_factory=tuple)


class MarketAdapter(ABC):
    info: MarketInfo

    # --- Symbols ---------------------------------------------------------------
    @abstractmethod
    def resolve(self, query: str) -> ResolvedSymbol:
        """Turn user input ('reliance', 'RELIANCE', 'Reliance Industries') into a symbol."""

    @abstractmethod
    def universe(self, name: str | None = None) -> pd.DataFrame:
        """Columns: symbol, ticker, name, industry."""

    def peers(self, resolved: ResolvedSymbol, k: int = 5) -> list[ResolvedSymbol]:
        """Same-industry peers from the default universe (used for valuation context)."""
        uni = self.universe()
        if not resolved.industry:
            return []
        same = uni[(uni["industry"] == resolved.industry) & (uni["symbol"] != resolved.symbol)]
        return [ResolvedSymbol(r.symbol, r.ticker, r.name, r.industry) for r in same.head(k).itertuples()]

    # --- Data ------------------------------------------------------------------
    @abstractmethod
    def history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Daily OHLCV indexed by date with columns Open, High, Low, Close, Volume."""

    @abstractmethod
    def history_bulk(self, tickers: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
        ...

    @abstractmethod
    def fundamentals(self, ticker: str) -> dict:
        """Normalised keys: pe, forward_pe, pb, roe, debt_to_equity, revenue_growth,
        earnings_growth, profit_margin, market_cap, sector, industry, name, dividend_yield."""

    @abstractmethod
    def news(self, resolved: ResolvedSymbol, lookback_days: int = 3, max_items: int = 25) -> list[NewsItem]:
        ...

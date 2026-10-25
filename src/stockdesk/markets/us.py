"""US market adapter - provision for a future release.

What it takes to switch this on (agents and graph need no changes):
  1. universe():  S&P 500 / NASDAQ-100 constituents (e.g. a bundled CSV refreshed from a
                  public constituents list) with columns symbol, ticker, name, industry.
  2. resolve():   map 'apple' / 'AAPL' to ResolvedSymbol; ticker == symbol (no suffix).
  3. history*/fundamentals(): reuse stockdesk.data.yahoo as-is (plain tickers work).
     For US filings, SEC EDGAR's free companyfacts API can enrich fundamentals.
  4. news():      reuse data.news.google_news(query, days, hl="en-US", gl="US") + yahoo.news.
  5. Register "US" in markets/registry.py and set MARKET=US in .env.
  6. Schedule: US close is 16:00 America/New_York (01:30/02:30 IST next day).
"""
from __future__ import annotations

from datetime import time

from stockdesk.markets.base import MarketAdapter, MarketInfo


class USAdapter(MarketAdapter):
    info = MarketInfo(
        key="US",
        display_name="US equities (NYSE / NASDAQ)",
        currency="USD",
        currency_symbol="$",
        timezone="America/New_York",
        close_time=time(16, 0),
        index_ticker="^GSPC",
        index_name="S&P 500",
        value_unit="M",
        value_divisor=1e6,
        universes=("sp500", "nasdaq100"),
    )

    def _todo(self, *_, **__):
        raise NotImplementedError("US market support is a planned extension - see the module docstring.")

    resolve = universe = history = history_bulk = fundamentals = news = _todo  # type: ignore[assignment]

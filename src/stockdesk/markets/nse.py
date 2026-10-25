"""NSE (India) market adapter.

Sources (all free):
  * Index constituents  - niftyindices.com CSVs (bundled NIFTY 50 snapshot as a fallback)
  * Full equity listing - NSE EQUITY_L.csv, used to resolve any listed symbol
  * Prices/fundamentals - Yahoo Finance via yfinance (".NS" tickers)
  * News                - Google News RSS (en-IN) + Yahoo Finance
"""
from __future__ import annotations

import io
import logging
from datetime import time
from difflib import SequenceMatcher
from importlib import resources

import httpx
import pandas as pd

from stockdesk.data import news as news_src
from stockdesk.data import yahoo
from stockdesk.data.cache import cached
from stockdesk.markets.base import (AmbiguousSymbol, MarketAdapter, MarketInfo, NewsItem,
                                    ResolvedSymbol, SymbolNotFound)

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

_UNIVERSE_URLS = {
    "nifty50": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "nifty100": "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "nifty200": "https://niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "nifty500": "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}
_EQUITY_LIST_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]

# Common shorthand people type -> NSE symbol
_ALIASES = {
    "L&T": "LT", "LARSEN": "LT", "SBI": "SBIN", "AIRTEL": "BHARTIARTL", "HUL": "HINDUNILVR",
    "M&M": "M&M", "MAHINDRA": "M&M", "ZOMATO": "ETERNAL", "BAJAJ AUTO": "BAJAJ-AUTO",
    "TATA MOTORS": "TMPV", "TATAMOTORS": "TMPV", "INFOSYS": "INFY", "KOTAK": "KOTAKBANK",
    "ULTRATECH": "ULTRACEMCO", "INDIGO": "INDIGO", "ICICI": "ICICIBANK", "HDFC": "HDFCBANK",
}


def _to_ticker(symbol: str) -> str:
    return f"{symbol}.NS"


class NSEAdapter(MarketAdapter):
    info = MarketInfo(
        key="NSE",
        display_name="National Stock Exchange of India",
        currency="INR",
        currency_symbol="₹",
        timezone="Asia/Kolkata",
        close_time=time(15, 30),
        index_ticker="^NSEI",
        index_name="NIFTY 50",
        value_unit="cr",
        value_divisor=1e7,
        universes=tuple(_UNIVERSE_URLS),
        global_benchmarks=(("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^VIX", "VIX")),
    )

    def __init__(self, default_universe: str = "nifty200"):
        self.default_universe = default_universe

    # --- Universe ----------------------------------------------------------------
    @staticmethod
    def _fallback() -> pd.DataFrame:
        raw = resources.files("stockdesk.markets.data").joinpath("nse_nifty50_fallback.csv").read_text()
        return pd.read_csv(io.StringIO(raw))

    @cached("nse.universe", ttl_hours=24 * 7)
    def _download_universe(self, name: str) -> pd.DataFrame | None:
        url = _UNIVERSE_URLS.get(name)
        if not url:
            raise ValueError(f"Unknown universe '{name}'. Options: {', '.join(_UNIVERSE_URLS)}")
        try:
            r = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as exc:
            log.warning("Could not download %s constituents (%s); using bundled NIFTY 50 snapshot", name, exc)
            return None

    def universe(self, name: str | None = None) -> pd.DataFrame:
        name = (name or self.default_universe).lower()
        df = self._download_universe(name)
        if df is None or df.empty:
            df = self._fallback()
        df = df.rename(columns={"Company Name": "name", "Industry": "industry", "Symbol": "symbol"})
        df = df[["symbol", "name", "industry"]].dropna(subset=["symbol"]).copy()
        df["symbol"] = df["symbol"].str.strip().str.upper()
        df["ticker"] = df["symbol"].map(_to_ticker)
        return df.reset_index(drop=True)

    @cached("nse.equity_list", ttl_hours=24 * 7)
    def _equity_list(self) -> pd.DataFrame | None:
        for url in _EQUITY_LIST_URLS:
            try:
                r = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                df = df.rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "name"})
                return df[["symbol", "name"]]
            except Exception as exc:
                log.info("equity list %s unavailable: %s", url, exc)
        return None

    # --- Resolution -----------------------------------------------------------------
    def resolve(self, query: str) -> ResolvedSymbol:
        q = query.strip().upper().removesuffix(".NS")
        q = _ALIASES.get(q, q)
        uni = self.universe("nifty500") if self.default_universe != "nifty50" else self.universe()
        industry_by_symbol = dict(zip(uni["symbol"], uni["industry"]))

        listing = self._equity_list()
        book = pd.concat([uni[["symbol", "name"]], listing], ignore_index=True) if listing is not None else uni[["symbol", "name"]]
        book = book.drop_duplicates("symbol")

        exact = book[book["symbol"] == q]
        if not exact.empty:
            row = exact.iloc[0]
            return ResolvedSymbol(row.symbol, _to_ticker(row.symbol), row["name"], industry_by_symbol.get(row.symbol, ""))

        # Fuzzy match on company name
        ql = q.lower()
        scored = []
        for row in book.itertuples():
            nm = news_src.clean_company_name(str(row.name)).lower()
            score = 1.0 if nm.startswith(ql) else SequenceMatcher(None, ql, nm).ratio()
            if ql in nm:
                score = max(score, 0.9)
            scored.append((score, row.symbol, row.name))
        scored.sort(reverse=True)
        top = [s for s in scored if s[0] >= 0.75][:6]
        if len(top) == 1 or (top and top[0][0] - (top[1][0] if len(top) > 1 else 0) >= 0.15):
            _, sym, nm = top[0]
            return ResolvedSymbol(sym, _to_ticker(sym), nm, industry_by_symbol.get(sym, ""))
        if len(top) > 1:
            raise AmbiguousSymbol(query, [ResolvedSymbol(s, _to_ticker(s), n, industry_by_symbol.get(s, "")) for _, s, n in top])

        # Last resort: the user typed a valid symbol we don't have in any list
        if q.replace("-", "").replace("&", "").isalnum() and not yahoo.history(_to_ticker(q), "5d").empty:
            return ResolvedSymbol(q, _to_ticker(q), q, "")
        raise SymbolNotFound(f"Could not find an NSE stock matching '{query}'.")

    # --- Data -----------------------------------------------------------------------
    def history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        return yahoo.history(ticker, period)

    def history_bulk(self, tickers: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
        return yahoo.history_bulk(tickers, period)

    def fundamentals(self, ticker: str) -> dict:
        return yahoo.fundamentals(ticker)

    def news(self, resolved: ResolvedSymbol, lookback_days: int = 3, max_items: int = 25) -> list[NewsItem]:
        name = news_src.clean_company_name(resolved.name)
        items = news_src.google_news(f'"{name}" share OR stock OR NSE', lookback_days, hl="en-IN", gl="IN")
        items += yahoo.news(resolved.ticker)
        items = news_src.relevant(news_src.within(items, lookback_days), resolved.name, resolved.symbol)
        return news_src.dedupe(items)[:max_items]

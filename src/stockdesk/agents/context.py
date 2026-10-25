"""Everything the analysts need about one stock, gathered once (concurrently) by the orchestrator."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from stockdesk.config import get_settings
from stockdesk.markets.base import MarketAdapter, MarketInfo, NewsItem, ResolvedSymbol

log = logging.getLogger(__name__)


@dataclass
class StockContext:
    resolved: ResolvedSymbol
    market: MarketInfo
    horizon: str
    history: pd.DataFrame
    index_history: pd.DataFrame
    global_history: dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: dict = field(default_factory=dict)
    peer_fundamentals: list[dict] = field(default_factory=list)
    news: list[NewsItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    use_openai: bool = True

    @property
    def name(self) -> str:
        return self.fundamentals.get("name") or self.resolved.name

    @property
    def last_close(self) -> float:
        return float(self.history["Close"].iloc[-1]) if not self.history.empty else float("nan")

    def money(self, v: float) -> str:
        return f"{self.market.currency_symbol}{v:,.2f}"


def gather(adapter: MarketAdapter, resolved: ResolvedSymbol, horizon: str,
           history: pd.DataFrame | None = None, index_history: pd.DataFrame | None = None,
           peers: int = 5, use_openai: bool = True) -> StockContext:
    """Fetch prices, index, fundamentals, peers and news in parallel threads."""
    s = get_settings()
    errors: list[str] = []

    def safe(fn, *a, default=None, label=""):
        try:
            return fn(*a)
        except Exception as exc:  # one failing source must not sink the analysis
            errors.append(f"{label}: {exc.__class__.__name__}")
            log.warning("%s failed for %s: %s", label, resolved.symbol, exc)
            return default

    def peer_funds() -> list[dict]:
        out = []
        for p in adapter.peers(resolved, k=peers):
            f = adapter.fundamentals(p.ticker)
            if f:
                out.append(f | {"symbol": p.symbol})
        return out

    global_tickers = [ticker for ticker, _ in adapter.info.global_benchmarks]
    with ThreadPoolExecutor(max_workers=6) as pool:
        f_hist = pool.submit(safe, adapter.history, resolved.ticker, "1y", default=pd.DataFrame(), label="prices") \
            if history is None else None
        f_idx = pool.submit(safe, adapter.history, adapter.info.index_ticker, "1y", default=pd.DataFrame(), label="index") \
            if index_history is None else None
        f_fund = pool.submit(safe, adapter.fundamentals, resolved.ticker, default={}, label="fundamentals")
        f_peers = pool.submit(safe, peer_funds, default=[], label="peers")
        f_news = pool.submit(safe, adapter.news, resolved, s.news_lookback_days, s.news_max_items, default=[], label="news")
        f_global = (pool.submit(safe, adapter.history_bulk, global_tickers, "3mo", default={}, label="global markets")
                    if global_tickers else None)

        return StockContext(
            resolved=resolved,
            market=adapter.info,
            horizon=horizon,
            history=f_hist.result() if f_hist else history,
            index_history=f_idx.result() if f_idx else index_history,
            global_history=f_global.result() or {} if f_global else {},
            fundamentals=f_fund.result() or {},
            peer_fundamentals=f_peers.result() or [],
            news=f_news.result() or [],
            errors=errors,
            use_openai=use_openai,
        )

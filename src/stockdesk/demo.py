"""Offline demo market: fictional companies with synthetic prices, news and fundamentals.

Used by `stockdesk demo` and by the test suite. Company names are invented on purpose
so a preview can never be mistaken for a
real recommendation.
"""
from __future__ import annotations

import zlib
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from stockdesk.markets.base import (MarketAdapter, MarketInfo, NewsItem, ResolvedSymbol,
                                    SymbolNotFound)

# symbol, name, industry, story (drives drift, last-day move and headlines)
_COMPANIES = [
    ("AURGRID", "Aurora Grid Systems Ltd.", "Power", "breakout"),
    ("BANYAN", "Banyan Foods Ltd.", "Fast Moving Consumer Goods", "steady"),
    ("KESTREL", "Kestrel Motors Ltd.", "Automobile and Auto Components", "breakout"),
    ("LOTUSDATA", "Lotus Data Labs Ltd.", "Information Technology", "uptrend"),
    ("MONSOON", "Monsoon Agritech Ltd.", "Chemicals", "downtrend"),
    ("NILGIRI", "Nilgiri Tea Estates Ltd.", "Fast Moving Consumer Goods", "flat"),
    ("ORBIT", "Orbit Telecom Networks Ltd.", "Telecommunication", "uptrend"),
    ("PEACOCK", "Peacock Finance Ltd.", "Financial Services", "steady"),
    ("QUARTZ", "Quartz Semiconductors Ltd.", "Information Technology", "overheated"),
    ("RIVERBEND", "Riverbend Cement Ltd.", "Construction Materials", "flat"),
    ("SAFFRON", "Saffron Healthcare Ltd.", "Healthcare", "uptrend"),
    ("TEAKWOOD", "Teakwood Infra Projects Ltd.", "Construction", "breakout"),
    ("UMBRA", "Umbra Steel Works Ltd.", "Metals & Mining", "downtrend"),
    ("VERMILION", "Vermilion Paints Ltd.", "Consumer Durables", "steady"),
    ("WAVELINE", "Waveline Logistics Ltd.", "Services", "negative_news"),
    ("ZENITH", "Zenith Bank Ltd.", "Financial Services", "uptrend"),
    ("ARECA", "Areca Pharma Ltd.", "Healthcare", "flat"),
    ("CEDAR", "Cedar Power Transmission Ltd.", "Power", "uptrend"),
]

_HEADLINES = {
    "breakout": [
        "{n} bags Rs 1,850 crore order, shares hit record high",
        "{n} Q2 net profit jumps 31% on strong demand, beats estimates",
        "Brokerage upgrades {n} to buy, raises target price",
        "{n} announces capacity expansion at new plant",
    ],
    "uptrend": [
        "{n} revenue up 18% YoY as margins expand",
        "{n} wins multi-year contract from global client",
        "Analysts stay upbeat on {n} after investor meet",
    ],
    "steady": [
        "{n} declares interim dividend",
        "{n} management sees stable demand in second half",
    ],
    "flat": [
        "{n} to hold board meeting next week",
        "{n} shares trade flat amid mixed market",
    ],
    "overheated": [
        "{n} shares surge for sixth straight session, rally looks stretched",
        "{n} profit jumps 40%, valuations now rich say analysts",
    ],
    "downtrend": [
        "{n} Q2 profit falls 22%, misses estimates on weak demand",
        "Brokerage downgrades {n}, cuts target price",
        "{n} faces margin pressure as input costs rise",
    ],
    "negative_news": [
        "SEBI issues show cause notice to {n} over disclosure lapses",
        "{n} CFO resigns with immediate effect",
        "{n} shares slump after promoter stake sale via block deal",
        "{n} wins small order from state utility",
    ],
}

_STORY = {  # annual drift, vol, last-day return, last-day volume multiple
    "breakout": (0.35, 0.24, 0.042, 2.6),
    "uptrend": (0.30, 0.20, 0.016, 1.4),
    "steady": (0.10, 0.16, 0.006, 1.0),
    "flat": (0.00, 0.18, -0.002, 0.9),
    "overheated": (0.90, 0.30, 0.055, 2.2),
    "downtrend": (-0.30, 0.26, -0.021, 1.8),
    "negative_news": (0.05, 0.28, 0.028, 2.4),
}


class DemoAdapter(MarketAdapter):
    info = MarketInfo(key="NSE", display_name="NSE (demo data)", currency="INR", currency_symbol="₹",
                      timezone="Asia/Kolkata", close_time=time(15, 30), index_ticker="DEMOINDEX.DEMO",
                      index_name="DEMO 50", universes=("demo",))

    def __init__(self, as_of: str | pd.Timestamp | None = None, seed: int = 7, **_):
        self.as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
        while self.as_of.weekday() >= 5:
            self.as_of -= pd.Timedelta(days=1)
        self.seed = seed
        self._rows = {s: (n, ind, story) for s, n, ind, story in _COMPANIES}
        self._hist_cache: dict[str, pd.DataFrame] = {}

    @property
    def now(self) -> datetime:
        return datetime.combine(self.as_of.date(), time(11, 0), tzinfo=timezone.utc)  # 16:30 IST

    def universe(self, name: str | None = None) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": s, "ticker": f"{s}.DEMO", "name": n, "industry": ind}
                             for s, (n, ind, _) in self._rows.items()])

    def resolve(self, query: str) -> ResolvedSymbol:
        q = query.strip().upper().removesuffix(".DEMO")
        for s, (n, ind, _) in self._rows.items():
            if q == s or q.lower() in n.lower():
                return ResolvedSymbol(s, f"{s}.DEMO", n, ind)
        raise SymbolNotFound(f"'{query}' is not in the demo universe: {', '.join(self._rows)}")

    def _make(self, ticker: str) -> pd.DataFrame:
        sym = ticker.removesuffix(".DEMO")
        days = pd.bdate_range(end=self.as_of, periods=260)
        rng = np.random.default_rng(zlib.crc32(f"{sym}:{self.seed}".encode()))
        if sym == "DEMOINDEX":
            drift, vol, last_ret, vol_mult, base = 0.12, 0.12, 0.0062, 1.1, 24000.0
        else:
            story = self._rows[sym][2]
            drift, vol, last_ret, vol_mult = _STORY[story]
            base = float(rng.uniform(150, 3500))
        rets = rng.normal(drift / 252, vol / np.sqrt(252), len(days))
        rets[-1] = last_ret
        if sym != "DEMOINDEX" and self._rows[sym][2] == "negative_news":
            rets[-15:-1] = rng.normal(-0.004, 0.015, 14)  # weak run-up, then a bounce on the last day
        close = base * np.exp(np.cumsum(rets))
        spread = np.abs(rng.normal(0, vol / np.sqrt(252) * 0.6, len(days)))
        high = close * (1 + spread)
        low = close * (1 - spread)
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, len(days)))
        volume = rng.lognormal(mean=14.2, sigma=0.25, size=len(days))
        volume[-1] = volume[-21:-1].mean() * vol_mult
        return pd.DataFrame({"Open": open_, "High": np.maximum(high, close), "Low": np.minimum(low, close),
                             "Close": close, "Volume": volume.round()}, index=days)

    def history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        if ticker not in self._hist_cache:
            self._hist_cache[ticker] = self._make(ticker)
        return self._hist_cache[ticker].copy()

    def history_bulk(self, tickers: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
        return {t: self.history(t, period) for t in tickers}

    def fundamentals(self, ticker: str) -> dict:
        sym = ticker.removesuffix(".DEMO")
        if sym not in self._rows:
            return {}
        n, ind, story = self._rows[sym]
        rng = np.random.default_rng(zlib.crc32(f"{sym}:f:{self.seed}".encode()))
        good = story in ("breakout", "uptrend")
        bad = story in ("downtrend",)
        return {
            "name": n, "sector": ind, "industry": ind,
            "market_cap": float(rng.uniform(2e10, 9e11)),
            "pe": float(rng.uniform(14, 28) if good else rng.uniform(30, 55) if story == "overheated" else rng.uniform(18, 40)),
            "forward_pe": None,
            "pb": float(rng.uniform(1.5, 8)),
            "roe": float(rng.uniform(0.17, 0.27) if good else rng.uniform(0.04, 0.10) if bad else rng.uniform(0.10, 0.18)),
            "debt_to_equity": float(rng.uniform(0.1, 0.6) if not bad else rng.uniform(1.1, 2.2)),
            "revenue_growth": float(rng.uniform(0.12, 0.25) if good else rng.uniform(-0.08, 0.02) if bad else rng.uniform(0.03, 0.12)),
            "earnings_growth": float(rng.uniform(0.15, 0.35) if good else rng.uniform(-0.25, -0.05) if bad else rng.uniform(0.0, 0.15)),
            "profit_margin": float(rng.uniform(0.12, 0.22) if good else rng.uniform(0.02, 0.06) if bad else rng.uniform(0.06, 0.14)),
            "dividend_yield": float(rng.uniform(0.2, 2.0)),
            "peg": None,
        }

    def news(self, resolved: ResolvedSymbol, lookback_days: int = 3, max_items: int = 25) -> list[NewsItem]:
        n, _, story = self._rows[resolved.symbol]
        short = n.replace(" Ltd.", "")
        out = []
        for i, h in enumerate(_HEADLINES[story]):
            out.append(NewsItem(title=h.format(n=short), link=f"https://example.com/demo/{resolved.symbol.lower()}/{i}",
                                source="Demo Wire", published=self.now - timedelta(hours=6 + 14 * i)))
        return out[:max_items]

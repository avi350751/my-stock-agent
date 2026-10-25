"""Yahoo Finance access via the open-source `yfinance` library.

Shared by every market adapter (NSE uses the .NS suffix; a US adapter would use
plain tickers), so the normalisation lives in one place.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from stockdesk.data.cache import cached
from stockdesk.markets.base import NewsItem

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1) if "Close" in df.columns.get_level_values(-1) else df.columns.get_level_values(0)
    df = df[[c for c in OHLCV if c in df.columns]]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
    return df.sort_index()


@cached("yahoo.history", ttl_hours=2)
def history(ticker: str, period: str = "1y") -> pd.DataFrame:
    import yfinance as yf
    try:
        raw = _with_retry(lambda: yf.Ticker(ticker).history(period=period, auto_adjust=True, actions=False),
                          f"history {ticker}", attempts=2)
    except Exception as exc:  # network / delisted / rate limit
        log.warning("history(%s) failed: %s", ticker, exc)
        return pd.DataFrame(columns=OHLCV)
    return _normalise(raw)


def _with_retry(fn, label: str, attempts: int = 3, base_delay: float = 2.0):
    """Yahoo rate-limits bursts; back off and retry instead of failing the whole run."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise
            delay = base_delay * (2 ** i)
            log.info("%s failed (%s); retrying in %.0fs", label, exc.__class__.__name__, delay)
            time.sleep(delay)


def history_bulk(tickers: list[str], period: str = "6mo", chunk: int = 50) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        try:
            raw = _with_retry(lambda: yf.download(batch, period=period, group_by="ticker", auto_adjust=True,
                                                  threads=True, progress=False), f"bulk download {i // chunk + 1}")
        except Exception as exc:
            log.warning("bulk download failed for %d tickers: %s", len(batch), exc)
            continue
        if i + chunk < len(tickers):
            time.sleep(1.0)  # be gentle with the free endpoint
        for t in batch:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = _normalise(sub)
            if len(df) >= 30:
                out[t] = df
    return out


def _pct_or_none(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


@cached("yahoo.fundamentals", ttl_hours=24)
def fundamentals(ticker: str) -> dict:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.warning("fundamentals(%s) failed: %s", ticker, exc)
        return {}
    dte = _pct_or_none(info.get("debtToEquity"))
    return {
        "name": info.get("longName") or info.get("shortName") or "",
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "market_cap": _pct_or_none(info.get("marketCap")),
        "pe": _pct_or_none(info.get("trailingPE")),
        "forward_pe": _pct_or_none(info.get("forwardPE")),
        "pb": _pct_or_none(info.get("priceToBook")),
        "roe": _pct_or_none(info.get("returnOnEquity")),
        # Yahoo reports D/E as a percentage (35.2 == 0.352x)
        "debt_to_equity": dte / 100 if dte is not None else None,
        "revenue_growth": _pct_or_none(info.get("revenueGrowth")),
        "earnings_growth": _pct_or_none(info.get("earningsGrowth")),
        "profit_margin": _pct_or_none(info.get("profitMargins")),
        "dividend_yield": _pct_or_none(info.get("dividendYield")),
        "peg": _pct_or_none(info.get("trailingPegRatio") or info.get("pegRatio")),
    }


def news(ticker: str, max_items: int = 15) -> list[NewsItem]:
    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        log.warning("yahoo news(%s) failed: %s", ticker, exc)
        return []
    items: list[NewsItem] = []
    for r in raw[:max_items]:
        c = r.get("content", r)  # yfinance >= 0.2.50 nests under "content"
        title = c.get("title") or ""
        if not title:
            continue
        link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link", "")
        provider = c.get("provider")
        source = provider.get("displayName", "") if isinstance(provider, dict) else c.get("publisher", "")
        published = None
        if c.get("pubDate"):
            try:
                published = datetime.fromisoformat(str(c["pubDate"]).replace("Z", "+00:00"))
            except ValueError:
                published = None
        elif c.get("providerPublishTime"):
            published = datetime.fromtimestamp(int(c["providerPublishTime"]), tz=timezone.utc)
        items.append(NewsItem(title=title, link=link or "", source=source or "Yahoo Finance",
                              published=published, summary=c.get("summary", "") or ""))
    return items

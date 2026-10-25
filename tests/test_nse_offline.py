"""NSE adapter and data-source parsing, exercised without network access."""
import httpx
import numpy as np
import pandas as pd
import pytest

from stockdesk.data import news as news_src
from stockdesk.data import yahoo
from stockdesk.markets.base import AmbiguousSymbol, SymbolNotFound
from stockdesk.markets.nse import NSEAdapter

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Google News</title>
<item><title>Infosys wins $1.5 bn deal from European bank - Economic Times</title>
<link>https://news.example/1</link><pubDate>Fri, 23 Oct 2026 08:10:00 GMT</pubDate>
<source url="https://economictimes.indiatimes.com">Economic Times</source>
<description>&lt;a href="x"&gt;Infosys wins deal&lt;/a&gt;</description></item>
<item><title>Nifty IT index ends higher - Mint</title>
<link>https://news.example/2</link><pubDate>Fri, 23 Oct 2026 09:00:00 GMT</pubDate>
<source url="https://livemint.com">Mint</source></item>
</channel></rss>"""


@pytest.fixture
def offline(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", boom)
    monkeypatch.setattr(yahoo, "history", lambda t, p="1y": pd.DataFrame())
    return NSEAdapter(default_universe="nifty200")


def test_universe_falls_back_to_bundled_list(offline):
    uni = offline.universe()
    assert len(uni) == 50 and uni["ticker"].str.endswith(".NS").all()
    assert {"symbol", "ticker", "name", "industry"} <= set(uni.columns)


@pytest.mark.parametrize("query,symbol", [
    ("RELIANCE", "RELIANCE"), ("reliance.ns", "RELIANCE"), ("infosys", "INFY"), ("SBI", "SBIN"),
    ("L&T", "LT"), ("Larsen", "LT"), ("hindustan unilever", "HINDUNILVR"), ("zomato", "ETERNAL"),
])
def test_resolve(offline, query, symbol):
    r = offline.resolve(query)
    assert r.symbol == symbol and r.ticker == f"{symbol}.NS"


def test_resolve_ambiguous_and_missing(offline):
    with pytest.raises(AmbiguousSymbol) as exc:
        offline.resolve("tata")
    assert len(exc.value.candidates) >= 2
    with pytest.raises(SymbolNotFound):
        offline.resolve("definitely not a company")


def test_peers_same_industry(offline):
    r = offline.resolve("HDFCBANK")
    peers = offline.peers(r, k=5)
    assert peers and all(p.industry == "Financial Services" and p.symbol != "HDFCBANK" for p in peers)


def test_yahoo_normalise_handles_tz_and_multiindex():
    idx = pd.date_range("2026-10-01", periods=5, freq="B", tz="Asia/Kolkata")
    raw = pd.DataFrame({("INFY.NS", c): np.arange(5) + 1.0 for c in ["Open", "High", "Low", "Close", "Volume"]}, index=idx)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    out = yahoo._normalise(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.tz is None and out.index[0] == pd.Timestamp("2026-10-01")


def test_google_news_parsing(monkeypatch):
    import feedparser
    real = feedparser.parse
    monkeypatch.setattr(feedparser, "parse", lambda url: real(RSS))
    items = news_src.google_news('"Infosys" share', 3)
    assert items[0].title == "Infosys wins $1.5 bn deal from European bank"   # publisher suffix stripped
    assert items[0].source == "Economic Times" and items[0].published.year == 2026
    assert [i.title for i in news_src.relevant(items, "Infosys Ltd.", "INFY")] == [items[0].title]


def test_yahoo_news_parsing_new_format(monkeypatch):
    class T:
        news = [{"id": "1", "content": {"title": "Infosys raises guidance", "pubDate": "2026-10-23T05:00:00Z",
                                        "canonicalUrl": {"url": "https://finance.yahoo.com/x"},
                                        "provider": {"displayName": "Reuters"}, "summary": "s"}}]
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda t: T())
    items = yahoo.news("INFY.NS")
    assert items[0].source == "Reuters" and items[0].link.endswith("/x") and items[0].published.tzinfo


def test_fundamentals_normalisation(monkeypatch):
    class T:
        info = {"longName": "Infosys Limited", "trailingPE": 24.5, "returnOnEquity": 0.31,
                "debtToEquity": 9.4, "revenueGrowth": 0.06, "sector": "Technology"}
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda t: T())
    f = yahoo.fundamentals.__wrapped__("INFY.NS")
    assert f["pe"] == 24.5 and f["debt_to_equity"] == pytest.approx(0.094) and f["name"] == "Infosys Limited"

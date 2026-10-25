"""Free news sources: Google News RSS (per-company query) + Yahoo Finance, de-duplicated."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.parse import quote_plus

from stockdesk.markets.base import NewsItem

log = logging.getLogger(__name__)

_SUFFIXES = re.compile(r"\b(limited|ltd\.?|inc\.?|corporation|corp\.?|plc|company)\b", re.I)


def clean_company_name(name: str) -> str:
    return re.sub(r"\s+", " ", _SUFFIXES.sub("", name)).strip(" .,&")


def google_news(query: str, lookback_days: int, hl: str = "en-IN", gl: str = "IN") -> list[NewsItem]:
    import feedparser
    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}+when:{lookback_days}d"
           f"&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}")
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        log.warning("google news failed for %r: %s", query, exc)
        return []
    out = []
    for e in feed.entries:
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        title = e.get("title", "")
        source = ""
        src = e.get("source")
        if isinstance(src, dict):
            source = src.get("title", "")
        # Google appends " - Publisher" to titles
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        out.append(NewsItem(title=title.strip(), link=e.get("link", ""), source=source, published=published,
                            summary=re.sub("<[^>]+>", " ", e.get("summary", ""))[:400]))
    return out


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def dedupe(items: list[NewsItem], threshold: float = 0.82) -> list[NewsItem]:
    kept: list[NewsItem] = []
    for it in sorted(items, key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
        n = _norm(it.title)
        if not n:
            continue
        if any(SequenceMatcher(None, n, _norm(k.title)).ratio() >= threshold for k in kept):
            continue
        kept.append(it)
    return kept


def within(items: list[NewsItem], lookback_days: int) -> list[NewsItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    return [i for i in items if i.published is None or i.published >= cutoff]


def relevant(items: list[NewsItem], name: str, symbol: str) -> list[NewsItem]:
    """Keep headlines that actually mention the company (Google search is fuzzy)."""
    key = clean_company_name(name).lower()
    words = key.split()
    patterns = {key} if key else set()
    if len(words) >= 2:
        patterns.add(" ".join(words[:2]))
    if words and len(words[0]) >= 6:
        patterns.add(words[0])
    if len(symbol) >= 3:
        patterns.add(symbol.lower())
    regexes = [re.compile(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])") for p in patterns]
    out = []
    for i in items:
        t = f"{i.title} {i.summary}".lower()
        if any(r.search(t) for r in regexes):
            out.append(i)
    return out

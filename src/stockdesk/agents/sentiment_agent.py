"""Sentiment analyst: recency-weighted tone of recent coverage."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from stockdesk.agents.context import StockContext
from stockdesk.analysis.indicators import clamp
from stockdesk.analysis.sentiment import get_scorer
from stockdesk.config import get_settings
from stockdesk.models import AgentReport, Evidence, Signal

HALF_LIFE_H = 24.0


def _age_hours(published: datetime | None, now: datetime) -> float:
    if published is None:
        return 48.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return max(0.0, (now - published).total_seconds() / 3600)


def run(ctx: StockContext, now: datetime | None = None) -> AgentReport:
    items = ctx.news
    if not items:
        return AgentReport(agent="sentiment", score=0.0, confidence=0.1, data_ok=False,
                           rationale="No recent coverage found, so sentiment is neutral by default.",
                           warnings=["No headlines in the lookback window"])

    now = now or datetime.now(timezone.utc)
    scorer = get_scorer()
    texts = [f"{i.title}. {i.summary[:200]}".strip() for i in items]
    scores = np.array(scorer.score(texts), dtype=float)
    ages = np.array([_age_hours(i.published, now) for i in items])
    # relative ages keep the weighted mean identical while avoiding underflow on old items
    weights = 0.5 ** ((ages - ages.min()) / HALF_LIFE_H)

    score = float(np.average(scores, weights=weights))
    dispersion = float(np.std(scores))
    recent = scores[ages <= 24]
    older = scores[ages > 24]
    shift = float(recent.mean() - older.mean()) if len(recent) and len(older) else 0.0

    n = len(items)
    confidence = min(1.0, n / 10) * (1 - min(0.5, dispersion / 2))
    confidence *= 0.8  # deterministic lexicon is intentionally conservative
    pos = int((scores > 0.15).sum())
    neg = int((scores < -0.15).sum())

    signals = [
        Signal(name="Weighted tone", value=f"{score:+.2f}", impact=clamp(score), note=scorer.name),
        Signal(name="Headline mix", value=f"{pos} positive / {neg} negative / {n - pos - neg} neutral",
               impact=clamp((pos - neg) / max(n, 1))),
        Signal(name="Coverage (buzz)", value=f"{n} articles in {get_settings().news_lookback_days}d", impact=0.0),
    ]
    if abs(shift) >= 0.15:
        signals.append(Signal(name="Tone shift (24h vs older)", value=f"{shift:+.2f}", impact=clamp(shift),
                              note="improving" if shift > 0 else "deteriorating"))

    order = np.argsort(-np.abs(scores * weights))
    evidence = [Evidence(title=f"{items[k].title} [{scores[k]:+.2f}]", url=items[k].link,
                         source=items[k].source, published=items[k].published) for k in order[:4]]

    mood = "positive" if score > 0.15 else "negative" if score < -0.15 else "mixed/neutral"
    rationale = (f"Coverage tone is {mood} ({score:+.2f}) across {n} recent articles "
                 f"({pos} positive, {neg} negative), scored with {scorer.name}.")
    return AgentReport(agent="sentiment", score=clamp(score), confidence=round(confidence, 3),
                       signals=signals, evidence=evidence, rationale=rationale)

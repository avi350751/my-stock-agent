"""News analyst: detects material corporate events and their likely price impact.

OpenAI classifies headlines when configured; a rule-based classifier is always
run and used as the fallback (and to sanity-check the model's output).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from stockdesk.agents.context import StockContext
from stockdesk.analysis.indicators import clamp
from stockdesk.config import get_settings
from stockdesk.llm.client import get_llm
from stockdesk.models import AgentReport, Evidence, Signal

log = logging.getLogger(__name__)

LABELS = {"earnings": "Earnings", "guidance": "Guidance", "order_win": "Order win", "rating_change": "Broker rating",
          "regulatory_legal": "Regulatory/legal", "management": "Management change", "corporate_action": "Corporate action",
          "mna_stake": "M&A / stake", "capex_expansion": "Capex/expansion", "promoter_activity": "Promoter activity",
          "macro_sector": "Macro/sector", "other": "Other"}

EventType = Literal["earnings", "guidance", "order_win", "rating_change", "regulatory_legal", "management",
                    "corporate_action", "mna_stake", "capex_expansion", "promoter_activity", "macro_sector", "other"]

# (event_type, pattern, impact, materiality)
_RULES: list[tuple[str, re.Pattern, float, float]] = [
    ("earnings", re.compile(r"(profit|pat|ebitda|revenue|earnings).{0,40}(jump|surge|rise|grow|up|beat|record)", re.I), 0.6, 0.9),
    ("earnings", re.compile(r"(profit|pat|ebitda|revenue|earnings).{0,40}(fall|drop|decline|down|miss|slump|plunge)", re.I), -0.6, 0.9),
    ("earnings", re.compile(r"\b(q[1-4]|quarter(ly)?)\b.{0,30}\bresults?\b", re.I), 0.0, 0.6),
    ("guidance", re.compile(r"guidance.{0,30}(raise|upgrade|improve|higher)", re.I), 0.5, 0.8),
    ("guidance", re.compile(r"guidance.{0,30}(cut|lower|reduce|withdraw)", re.I), -0.6, 0.8),
    ("order_win", re.compile(r"(bags|wins|secures|receives|bagged|won).{0,40}(order|contract|deal|project)", re.I), 0.45, 0.7),
    ("rating_change", re.compile(r"(upgrade[sd]?|raises? target|target (price )?(raised|hiked)|initiates? .{0,20}buy|outperform)", re.I), 0.45, 0.6),
    ("rating_change", re.compile(r"(downgrade[sd]?|cuts? target|target (price )?(cut|lowered)|underperform|\bsell rating)", re.I), -0.45, 0.6),
    ("regulatory_legal", re.compile(r"\b(sebi|rbi|cci|usfda|fda|income tax|ed|cbi)\b.{0,50}(notice|probe|penalty|order|raid|warning|ban|observations?)", re.I), -0.55, 0.8),
    ("regulatory_legal", re.compile(r"(approval|clearance|nod)\b.{0,30}\b(usfda|fda|sebi|rbi|cci|government)", re.I), 0.4, 0.6),
    ("management", re.compile(r"\b(ceo|md|cfo|chairman|director)\b.{0,30}\b(resigns?|quits?|steps down|exit)", re.I), -0.35, 0.6),
    ("management", re.compile(r"\b(appoints?|names?)\b.{0,30}\b(ceo|md|cfo|chairman)\b", re.I), 0.1, 0.4),
    ("corporate_action", re.compile(r"\b(buyback|bonus issue|stock split|special dividend|dividend)\b", re.I), 0.3, 0.5),
    ("mna_stake", re.compile(r"\b(acquires?|acquisition|merger|amalgamation|takeover|demerger)\b", re.I), 0.2, 0.6),
    ("promoter_activity", re.compile(r"promoter.{0,30}(pledge|sell|offload|stake sale)|block deal|ofs\b", re.I), -0.35, 0.6),
    ("promoter_activity", re.compile(r"promoter.{0,30}(buy|raise|increase).{0,15}stake", re.I), 0.35, 0.6),
    ("capex_expansion", re.compile(r"(capacity expansion|new plant|commissions?|capex|expands? capacity)", re.I), 0.3, 0.5),
]


class LLMEvent(BaseModel):
    index: int = Field(ge=0)
    event_type: EventType
    impact: float = Field(ge=-1, le=1, description="likely short-term price impact, -1..+1")
    materiality: float = Field(ge=0, le=1, description="how much this matters to valuation, 0..1")
    why: str = Field(max_length=200)


class LLMEvents(BaseModel):
    events: list[LLMEvent]


_SYSTEM = ("You are an equity news analyst for the Indian stock market. Classify each headline about the "
           "given company. Be conservative: routine or speculative stories have materiality below 0.3. "
           "Never invent facts beyond the headline.")


def classify_rules(title: str) -> tuple[str, float, float]:
    best = ("other", 0.0, 0.0)
    for ev, pat, impact, materiality in _RULES:
        if pat.search(title) and materiality > best[2]:
            best = (ev, impact, materiality)
    return best


def classify_llm(ctx: StockContext, titles: list[str]) -> dict[int, LLMEvent]:
    if not ctx.use_openai:
        return {}
    llm = get_llm()
    if not llm.available or not titles:
        return {}
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
    user = f"Company: {ctx.name} ({ctx.resolved.symbol}, {ctx.market.key})\nHeadlines:\n{listing}"
    res = llm.complete_json(_SYSTEM, user, LLMEvents)
    if res is None:
        return {}
    return {e.index: e for e in res.events if e.index < len(titles)}


def run(ctx: StockContext, now: datetime | None = None) -> AgentReport:
    items = ctx.news[:15]
    if not items:
        return AgentReport(agent="news", score=0.0, confidence=0.2, data_ok=False,
                           rationale="No news in the lookback window - no event risk detected, but also no catalyst.",
                           warnings=["No headlines in the lookback window"])
    now = now or datetime.now(timezone.utc)
    titles = [i.title for i in items]
    llm_events = classify_llm(ctx, titles)
    used_llm = bool(llm_events)

    material = []
    for k, it in enumerate(items):
        ev_type, impact, materiality = classify_rules(it.title)
        source = "rules"
        if k in llm_events:
            e = llm_events[k]
            # Guard: if rules and LLM disagree in direction on a high-materiality event, trust neither fully.
            if ev_type != "other" and impact * e.impact < 0 and materiality >= 0.6:
                impact, materiality, source = (impact + e.impact) / 2, min(materiality, e.materiality), "rules+llm (disputed)"
            else:
                ev_type, impact, materiality, source = e.event_type, e.impact, e.materiality, "llm"
        if materiality < 0.3 or ev_type == "other":
            continue
        pub = it.published or now
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_h = max(0.0, (now - pub).total_seconds() / 3600)
        material.append([it, ev_type, impact, materiality, age_h, source])

    if not material:
        return AgentReport(agent="news", score=0.0, confidence=0.35,
                           signals=[Signal(name="Material events", value="none", impact=0.0)],
                           evidence=[Evidence(title=i.title, url=i.link, source=i.source, published=i.published) for i in items[:3]],
                           rationale=f"Scanned {len(items)} headlines; nothing material (routine coverage only).")

    newest = min(m[4] for m in material)
    for m in material:  # half-life 36h, relative to the newest event (same mean, no underflow)
        m[4] = 0.5 ** ((m[4] - newest) / 36)
    num = sum(imp * mat * rec for _, _, imp, mat, rec, _ in material)
    den = sum(mat * rec for _, _, _, mat, rec, _ in material)
    score = clamp(num / den if den else 0.0)
    confidence = min(1.0, 0.4 + 0.15 * len(material)) * (1.0 if used_llm else 0.85)

    material.sort(key=lambda x: abs(x[2]) * x[3] * x[4], reverse=True)
    signals = [Signal(name=LABELS.get(ev, ev), value=f"{imp:+.2f} impact, materiality {mat:.1f}",
                      impact=clamp(imp), note=it.title[:120]) for it, ev, imp, mat, _, _ in material[:6]]
    evidence = [Evidence(title=it.title, url=it.link, source=it.source, published=it.published)
                for it, *_ in material[:5]]
    top = material[0]
    rationale = (f"{len(material)} material event(s) in the last {get_settings().news_lookback_days} days; the most important is "
                 f"{LABELS.get(top[1], top[1]).lower()} ({top[2]:+.2f}). Net event impact {score:+.2f}"
                 f" (classified by {'LLM + rules' if used_llm else 'rules'}).")
    return AgentReport(agent="news", score=score, confidence=round(confidence, 3), signals=signals,
                       evidence=evidence, rationale=rationale)

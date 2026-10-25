"""Render a single-stock analysis as a self-contained local HTML report."""
from __future__ import annotations

import re

from jinja2 import Environment, PackageLoader, select_autoescape

from stockdesk.analysis.sentiment import get_scorer
from stockdesk.config import HORIZON_WEIGHTS, get_settings
from stockdesk.markets.base import MarketInfo
from stockdesk.models import StockReport, Verdict

_env = Environment(loader=PackageLoader("stockdesk", "report/templates"),
                   autoescape=select_autoescape(["html", "j2"]))

VERDICT_STYLE = {
    Verdict.BUY: ("#15803D", "#DCFCE7"),
    Verdict.HOLD: ("#B7791F", "#FEF3C7"),
    Verdict.SELL: ("#B91C1C", "#FEE2E2"),
}
AGENT_LABEL = {
    "market": "Global & technical",
    "sentiment": "Investor sentiment",
    "news": "Company news",
    "fundamentals": "Fundamentals",
}


def _score_color(value: float) -> str:
    return "#15803D" if value >= 0.15 else "#C2410C" if value <= -0.15 else "#B7791F"


def _first_sentences(text: str, count: int = 2) -> str:
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[:count])


def _view(rep: StockReport, info: MarketInfo, chart_src: str | None) -> dict:
    snap, rec = rep.snapshot, rep.recommendation
    plan = rec.investor_call
    fg, bg = VERDICT_STYLE[rec.verdict]
    up = snap.change_pct >= 0
    weights = HORIZON_WEIGHTS.get(rep.horizon, HORIZON_WEIGHTS["positional"])
    agents = []
    for key in ("market", "sentiment", "news", "fundamentals"):
        report = rep.reports.get(key)
        if report:
            agents.append({"label": AGENT_LABEL[key], "score_str": f"{report.score:+.2f}",
                           "bar_pct": max(4, round((report.score + 1) / 2 * 100)),
                           "color": _score_color(report.score)})
    ranked = sorted(((signal, weights.get(agent, 0.2)) for agent, report in rep.reports.items()
                     for signal in report.signals), key=lambda item: item[0].impact * item[1], reverse=True)
    why = [f"{signal.name}: {signal.value}" for signal, _ in ranked if signal.impact > 0.15][:4]
    if not why:
        why = [f"{signal.name}: {signal.value}" for signal, _ in ranked[:3]]
    seen, sources = set(), []
    for key in ("news", "sentiment"):
        report = rep.reports.get(key)
        for evidence in (report.evidence if report else []):
            title = re.sub(r"\s*\[[+-]\d\.\d+\]$", "", evidence.title)
            if evidence.url and title not in seen:
                seen.add(title)
                sources.append({"title": title, "url": evidence.url, "source": evidence.source})
    return {
        "name": snap.name, "symbol": snap.symbol, "industry": snap.industry,
        "close_str": f"{info.currency_symbol}{snap.last_close:,.2f}",
        "change_str": f"{'▲' if up else '▼'} {abs(snap.change_pct):.2f}%",
        "change_fg": "#15803D" if up else "#B91C1C", "change_bg": "#DCFCE7" if up else "#FEE2E2",
        "verdict": rec.verdict.value, "verdict_fg": fg, "verdict_bg": bg,
        "composite_str": f"{rec.composite:+.2f}", "confidence_pct": round(rec.confidence * 100),
        "strength_pct": max(4, round((rec.composite + 1) / 2 * 100)), "agents": agents,
        "why": why, "flips": rec.flips_if, "flags": rec.rule_flags, "note": rec.note,
        "note_short": _first_sentences(rec.note), "sources": sources[:3], "chart_src": chart_src,
        "investor": {
            "posture": plan.posture,
            "buy_zone": (f"{info.currency_symbol}{plan.buy_zone_low:,.2f} – {info.currency_symbol}{plan.buy_zone_high:,.2f}"
                         if plan.buy_zone_low and plan.buy_zone_high else None),
            "trigger": f"{info.currency_symbol}{plan.trigger_price:,.2f}" if plan.trigger_price else None,
            "invalidation": (f"{info.currency_symbol}{plan.invalidation_price:,.2f}"
                             if plan.invalidation_price else None),
            "guidance": plan.guidance, "basis": plan.basis, "as_of": f"{snap.as_of:%d %b %Y}",
        },
    }


def render_analysis(rep: StockReport, info: MarketInfo, chart_src: str | None = None,
                    sample: bool = False) -> tuple[str, str]:
    view = _view(rep, info, chart_src)
    subject = f"Stock analysis · {rep.snapshot.as_of:%d %b %Y} · {view['name']} ({view['verdict']})"
    backend = get_scorer().name
    method = (f"Four specialist agents ran in parallel: global and technical markets, investor sentiment "
              f"({backend}), material company news, and fundamentals versus peers. The chief analyst combined "
              f"them with {rep.horizon}-horizon weights and hard risk rules.")
    html = _env.get_template("digest.html.j2").render(
        subject=subject, sample=sample, kicker="On-demand analysis", title="Stock analysis",
        market_label=info.key, date_long=f"{rep.snapshot.as_of:%d %B %Y} · {info.key} close",
        mood=None, hero=view, hero_label="Chief analyst decision", others=[], notes=[], method=method,
        preheader=f"{view['name']}: {view['verdict']} · {view['note_short'][:110]}",
    )
    return subject, html

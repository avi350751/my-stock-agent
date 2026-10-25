"""Fundamentals analyst: quality, leverage, growth and valuation vs same-industry peers."""
from __future__ import annotations

import statistics

from stockdesk.agents.context import StockContext
from stockdesk.analysis.indicators import clamp
from stockdesk.models import AgentReport, Signal

WEIGHTS = {"profitability": 0.25, "leverage": 0.15, "growth": 0.25, "valuation": 0.25, "margin": 0.10}


def _is_financial(ctx: StockContext) -> bool:
    text = f"{ctx.fundamentals.get('sector', '')} {ctx.resolved.industry}".lower()
    return "financial" in text or "bank" in text


def _median(peers: list[dict], key: str) -> float | None:
    vals = [p[key] for p in peers if isinstance(p.get(key), (int, float)) and p[key] > 0]
    return statistics.median(vals) if len(vals) >= 2 else None


def _band(x: float, bands: list[tuple[float, float]], default: float) -> float:
    """bands: list of (threshold, score) checked in order with x >= threshold."""
    for th, sc in bands:
        if x >= th:
            return sc
    return default


def run(ctx: StockContext) -> AgentReport:
    f = ctx.fundamentals
    if not f or all(f.get(k) is None for k in ("pe", "roe", "revenue_growth", "pb")):
        return AgentReport(agent="fundamentals", score=0.0, confidence=0.0, data_ok=False,
                           rationale="Fundamental data unavailable from the free source today.",
                           warnings=["No fundamentals returned"])

    sub: dict[str, float] = {}
    signals: list[Signal] = []
    pct = lambda v: f"{v * 100:.1f}%"

    roe = f.get("roe")
    if roe is not None:
        sub["profitability"] = _band(roe, [(0.20, 1.0), (0.15, 0.5), (0.10, 0.0), (0.0, -0.5)], -1.0)
        signals.append(Signal(name="ROE", value=pct(roe), impact=sub["profitability"]))

    dte = f.get("debt_to_equity")
    if dte is not None and not _is_financial(ctx):
        sub["leverage"] = 0.5 if dte < 0.5 else 0.0 if dte < 1 else -0.4 if dte < 2 else -0.8
        signals.append(Signal(name="Debt / equity", value=f"{dte:.2f}x", impact=sub["leverage"]))

    g_parts = []
    rg, eg = f.get("revenue_growth"), f.get("earnings_growth")
    if rg is not None:
        s = _band(rg, [(0.15, 0.6), (0.05, 0.2), (0.0, -0.1)], -0.6)
        g_parts.append(s)
        signals.append(Signal(name="Revenue growth (YoY)", value=pct(rg), impact=s))
    if eg is not None:
        s = _band(eg, [(0.20, 0.8), (0.08, 0.3), (0.0, -0.1)], -0.7)
        g_parts.append(s)
        signals.append(Signal(name="Earnings growth (YoY)", value=pct(eg), impact=s))
    if g_parts:
        sub["growth"] = sum(g_parts) / len(g_parts)

    pe = f.get("pe")
    peer_pe = _median(ctx.peer_fundamentals, "pe")
    if pe is not None:
        if pe <= 0:
            sub["valuation"] = -0.5
            signals.append(Signal(name="P/E", value="negative (loss-making)", impact=-0.5))
        elif peer_pe:
            ratio = pe / peer_pe
            sub["valuation"] = 0.5 if ratio < 0.8 else 0.0 if ratio <= 1.2 else -0.3 if ratio <= 1.6 else -0.6
            signals.append(Signal(name="P/E vs peers", value=f"{pe:.1f} vs median {peer_pe:.1f}",
                                  impact=sub["valuation"], note=f"{len(ctx.peer_fundamentals)} industry peers"))
        else:
            peg = f.get("peg")
            if peg and peg > 0:
                sub["valuation"] = 0.4 if peg < 1 else 0.0 if peg <= 2 else -0.4
                signals.append(Signal(name="PEG", value=f"{peg:.2f}", impact=sub["valuation"]))
            else:
                sub["valuation"] = 0.2 if pe < 15 else 0.0 if pe < 35 else -0.3 if pe < 60 else -0.5
                signals.append(Signal(name="P/E (absolute)", value=f"{pe:.1f}", impact=sub["valuation"],
                                      note="no peer data; absolute bands"))
    pb = f.get("pb")
    if pb is not None:
        peer_pb = _median(ctx.peer_fundamentals, "pb")
        signals.append(Signal(name="P/B", value=f"{pb:.1f}" + (f" vs median {peer_pb:.1f}" if peer_pb else ""), impact=0.0))

    pm = f.get("profit_margin")
    if pm is not None:
        sub["margin"] = 0.3 if pm > 0.15 else -0.3 if pm < 0.05 else 0.0
        signals.append(Signal(name="Net margin", value=pct(pm), impact=sub["margin"]))

    used = {k: w for k, w in WEIGHTS.items() if k in sub}
    score = sum(sub[k] * w for k, w in used.items()) / sum(used.values()) if used else 0.0
    applicable = sum(w for k, w in WEIGHTS.items() if not (k == "leverage" and _is_financial(ctx)))
    confidence = round(min(1.0, sum(used.values()) / applicable) * (1.0 if peer_pe else 0.85), 3)

    quality = "high" if sub.get("profitability", 0) >= 0.5 else "average" if sub.get("profitability", 0) >= 0 else "weak"
    val = sub.get("valuation", 0)
    val_word = "attractive" if val > 0.2 else "fair" if val > -0.2 else "rich"
    rationale = (f"Business quality looks {quality}"
                 + (f" (ROE {pct(roe)})" if roe is not None else "")
                 + f", growth is {'healthy' if sub.get('growth', 0) > 0.2 else 'modest' if sub.get('growth', 0) >= 0 else 'contracting'}"
                 + f" and valuation is {val_word}"
                 + (f" against {len(ctx.peer_fundamentals)} industry peers." if peer_pe else "."))
    signals.sort(key=lambda s: abs(s.impact), reverse=True)
    return AgentReport(agent="fundamentals", score=clamp(score), confidence=confidence, signals=signals,
                       rationale=rationale)

"""Chief analyst: merges the four reports into one call.

Two layers:
  1. Horizon-weighted, confidence-scaled composite score.
  2. Hard rules the LLM cannot override (signal conflict, data quality, liquidity,
     material negative news, overextension).
The LLM only writes the human-readable note.
"""
from __future__ import annotations

import numpy as np

from stockdesk.agents import market_agent
from stockdesk.agents.context import StockContext
from stockdesk.analysis import indicators as ta
from stockdesk.analysis.indicators import clamp
from stockdesk.config import HORIZON_WEIGHTS, get_settings
from stockdesk.llm.client import get_llm
from stockdesk.models import AgentReport, InvestorCall, Recommendation, Verdict

BUY_AT = 0.20
SELL_AT = -0.20
MIN_CONFIDENCE = 0.35


def combine(reports: dict[str, AgentReport], horizon: str) -> tuple[float, float]:
    weights = HORIZON_WEIGHTS[horizon]
    eff = {a: weights[a] * reports[a].confidence for a in weights if a in reports}
    total = sum(eff.values())
    composite = sum(eff[a] * reports[a].score for a in eff) / total if total > 1e-6 else 0.0

    informed = [reports[a].score for a in eff if reports[a].confidence >= 0.3]
    agreement = 1 - min(0.5, float(np.std(informed))) if len(informed) >= 2 else 0.7
    base_conf = sum(weights[a] * reports[a].confidence for a in weights if a in reports)
    return clamp(composite), round(max(0.0, min(1.0, base_conf * agreement)), 3)


def decide(ctx: StockContext, reports: dict[str, AgentReport]) -> Recommendation:
    s = get_settings()
    composite, confidence = combine(reports, ctx.horizon)
    flags: list[str] = []
    buy_vetoes: list[str] = []
    cap_to_hold = False

    # Rule 1: strong, confident disagreement between analysts -> no directional call
    strong_pos = [a for a, r in reports.items() if r.score >= 0.4 and r.confidence >= 0.6]
    strong_neg = [a for a, r in reports.items() if r.score <= -0.4 and r.confidence >= 0.6]
    if strong_pos and strong_neg:
        cap_to_hold = True
        flags.append(f"Conflicting signals: {', '.join(strong_pos)} bullish vs {', '.join(strong_neg)} bearish")

    # Rule 2: data quality
    missing = [a for a, r in reports.items() if not r.data_ok]
    if "market" in missing:
        cap_to_hold = True
        flags.append("No reliable price data - cannot make a directional call")
    if len(missing) >= 2:
        confidence = round(confidence * 0.6, 3)
        flags.append(f"Thin data from: {', '.join(missing)}")

    # Rule 3: liquidity
    if not ctx.history.empty:
        atv = ta.avg_traded_value(ctx.history) / ctx.market.value_divisor
        if atv < s.min_avg_traded_value_cr:
            buy_vetoes.append(f"Thin liquidity: avg traded value {ctx.market.currency_symbol}{atv:,.1f} {ctx.market.value_unit}/day")

    # Rule 4: material negative news vetoes a buy
    news = reports.get("news")
    if news and news.score <= -0.5 and news.confidence >= 0.5:
        buy_vetoes.append("Material negative news - no fresh buying until it is digested")

    # Rule 5: overextended
    lv = market_agent.levels(ctx) if not ctx.history.empty and len(ctx.history) >= 30 else {}
    if lv.get("rsi") and lv["rsi"] > 78:
        buy_vetoes.append(f"Overextended (RSI {lv['rsi']:.0f}) - better entries usually come on a pullback")

    if confidence < MIN_CONFIDENCE:
        cap_to_hold = True
        flags.append(f"Low overall confidence ({confidence:.2f})")

    if cap_to_hold:
        verdict = Verdict.HOLD
    elif composite >= BUY_AT and not buy_vetoes:
        verdict = Verdict.BUY
    elif composite <= SELL_AT:
        verdict = Verdict.SELL
    else:
        verdict = Verdict.HOLD

    if buy_vetoes:
        flags.extend(buy_vetoes)

    return Recommendation(verdict=verdict, composite=composite,
                          confidence=confidence, flips_if=_flips(ctx, verdict, lv), rule_flags=flags,
                          note=template_note(ctx, reports, verdict, composite),
                          investor_call=_investor_call(ctx, verdict, lv, buy_vetoes))


def _price(value: float | None) -> float | None:
    return round(float(value), 2) if value is not None and value > 0 else None


def _investor_call(ctx: StockContext, verdict: Verdict, lv: dict,
                   buy_vetoes: list[str]) -> InvestorCall:
    """Build an indicative entry plan without asking the LLM for price levels."""
    if not lv:
        return InvestorCall(posture="WAIT" if verdict is not Verdict.SELL else "AVOID",
                            guidance="No fresh entry until reliable price history is available.",
                            basis="Price data is insufficient to calculate a defensible entry level.")

    close = float(lv["close"])
    sma20 = lv.get("sma20")
    sma50 = lv.get("sma50")
    swing_low = lv.get("swing_low_20")
    swing_high = lv.get("swing_high_20")
    invalidation = _price((swing_low or sma50) * 0.99) if (swing_low or sma50) else None

    if verdict is Verdict.BUY:
        anchor = min(close, sma20) if sma20 else close
        low, high = _price(anchor * 0.98), _price(anchor * 1.01)
        inside = low <= close <= high
        guidance = ("Current price is inside the indicative zone; consider 2-3 staggered limit entries."
                    if inside else
                    "Wait for a pullback into the indicative zone and consider 2-3 staggered limit entries; do not chase above it.")
        return InvestorCall(posture="ACCUMULATE", buy_zone_low=low, buy_zone_high=high,
                            invalidation_price=invalidation, guidance=guidance,
                            basis="Zone is centred on the 20-day average (or current price when unavailable).")

    if verdict is Verdict.HOLD:
        if buy_vetoes and sma20:
            low, high = _price(sma20 * 0.98), _price(sma20 * 1.01)
            return InvestorCall(posture="WAIT", buy_zone_low=low, buy_zone_high=high,
                                invalidation_price=invalidation,
                                guidance="This is a watch zone, not an immediate buy. Enter only after the listed risk flags clear.",
                                basis="Positive evidence exists, but a risk veto requires a pullback or confirmation.")
        trigger = _price(swing_high * 1.01) if swing_high else None
        high = _price(trigger * 1.02) if trigger else None
        return InvestorCall(posture="WAIT", buy_zone_low=trigger, buy_zone_high=high,
                            trigger_price=trigger, invalidation_price=_price(sma50 or swing_low),
                            guidance="Consider a new entry only after a daily close above the trigger with stronger-than-average volume.",
                            basis="Mixed evidence requires a confirmed breakout above the 20-day range.")

    recovery = _price(max(v for v in (sma50, swing_high) if v is not None)) if (sma50 or swing_high) else None
    return InvestorCall(posture="AVOID", trigger_price=recovery,
                        guidance="No active buy zone while the call is SELL. Reassess only after a sustained recovery above the trigger.",
                        basis="The evidence is negative; capital preservation takes priority over finding an entry.")


def _flips(ctx: StockContext, verdict: Verdict, lv: dict) -> list[str]:
    if not lv:
        return ["Reliable price data becomes available"]
    m = ctx.money
    out = []
    if verdict is Verdict.BUY:
        if lv.get("sma50"):
            out.append(f"A close below the 50-DMA ({m(lv['sma50'])})")
        out.append(f"A break below the 20-day swing low ({m(lv['swing_low_20'])})")
        out.append("A material negative event (downgrade, regulatory action, weak results)")
    elif verdict is Verdict.SELL:
        if lv.get("sma50"):
            out.append(f"A close back above the 50-DMA ({m(lv['sma50'])})")
        out.append(f"A move above the 20-day high ({m(lv['swing_high_20'])}) on strong volume")
        out.append("A positive catalyst (results beat, upgrade, large order win)")
    else:
        out.append(f"Upside: a close above the 20-day high ({m(lv['swing_high_20'])}) with volume")
        out.append(f"Downside: a close below the 20-day low ({m(lv['swing_low_20'])})")
    return out


def _first(text: str) -> str:
    return text.split(". ")[0].rstrip(".") + "."


def template_note(ctx: StockContext, reports: dict[str, AgentReport], verdict: Verdict, composite: float) -> str:
    """Deterministic note used when no LLM is configured (and as the LLM's fallback)."""
    ranked = sorted(reports.values(), key=lambda r: r.score * max(r.confidence, 0.1), reverse=True)
    best, second, worst = ranked[0], ranked[1], ranked[-1]
    if verdict is Verdict.BUY:
        lead = (f"{best.agent.title()} ({best.score:+.2f}) and {second.agent} ({second.score:+.2f}) "
                f"lead a constructive case.")
    elif verdict is Verdict.SELL:
        lead = (f"{worst.agent.title()} ({worst.score:+.2f}) and {ranked[-2].agent} ({ranked[-2].score:+.2f}) "
                f"point the same way, so the case is negative.")
    else:
        lead = f"The analysts disagree, so this is a wait-and-see: {best.agent} {best.score:+.2f} vs {worst.agent} {worst.score:+.2f}."
    focus = worst if verdict is Verdict.SELL else best
    softest = min(reports.values(), key=lambda r: r.score)
    tail = (f"No analyst is negative; the softest read is {softest.agent} ({softest.score:+.2f})."
            if softest.score > 0.15 else f"Main risk: {_first(worst.rationale)}" if verdict is not Verdict.SELL
            else f"Best counterpoint: {_first(best.rationale)}")
    return f"{lead} {_first(focus.rationale)} {tail}"


_NOTE_SYSTEM = ("You are a calm, plain-spoken equity analyst writing for a retail investor in India. "
                "Write 3 short sentences explaining the call using ONLY the facts provided. No hype, no "
                "guarantees, no price targets, no numbers that are not in the input.")


def narrate(ctx: StockContext, reports: dict[str, AgentReport], rec: Recommendation) -> str:
    llm = get_llm()
    if not llm.available:
        return rec.note
    facts = [f"Stock: {ctx.name} ({ctx.resolved.symbol}), close {ctx.money(ctx.last_close)}",
             f"Call: {rec.verdict.value}",
             f"Composite {rec.composite:+.2f}, confidence {rec.confidence:.2f}, horizon {ctx.horizon}"]
    for r in reports.values():
        top = "; ".join(f"{sg.name}: {sg.value}" for sg in r.signals[:3])
        facts.append(f"{r.agent} score {r.score:+.2f} (conf {r.confidence:.2f}) - {r.rationale} Top signals: {top}")
    if rec.rule_flags:
        facts.append("Risk flags: " + " | ".join(rec.rule_flags))
    text = llm.complete(_NOTE_SYSTEM, "\n".join(facts), temperature=0.3)
    return text.strip() if text and len(text.strip()) > 40 else rec.note

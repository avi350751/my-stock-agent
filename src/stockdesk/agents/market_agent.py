"""Market analyst: global backdrop, trend, momentum, relative strength and volume."""
from __future__ import annotations

import numpy as np

from stockdesk.agents.context import StockContext
from stockdesk.analysis import indicators as ta
from stockdesk.models import AgentReport, Signal

WEIGHTS = {"global": 0.15, "trend": 0.30, "momentum": 0.20, "relative_strength": 0.17,
           "volume": 0.09, "range": 0.09}


def levels(ctx: StockContext) -> dict:
    """Key price levels, reused by the chief analyst for 'what would flip the call'."""
    c = ctx.history["Close"]
    out = {"close": float(c.iloc[-1]), "swing_low_20": ta.swing_low(c, 20), "swing_high_20": ta.swing_high(c, 20)}
    for n in (20, 50, 200):
        v = ta.sma(c, n).iloc[-1] if len(c) >= n else np.nan
        out[f"sma{n}"] = None if np.isnan(v) else float(v)
    r = ta.rsi(c).iloc[-1] if len(c) > 15 else np.nan
    out["rsi"] = None if np.isnan(r) else float(r)
    return out


def run(ctx: StockContext) -> AgentReport:
    df = ctx.history
    if df is None or len(df) < 30:
        return AgentReport(agent="market", score=0.0, confidence=0.0, data_ok=False,
                           rationale="Not enough price history to analyse.",
                           warnings=["Price history unavailable or shorter than 30 sessions"])

    c, v = df["Close"], df["Volume"]
    lv = levels(ctx)
    close = lv["close"]
    signals: list[Signal] = []
    sub: dict[str, float] = {}
    m = ctx.money

    # Global risk backdrop ------------------------------------------------------
    global_parts = []
    names = dict(ctx.market.global_benchmarks)
    for ticker, hist in ctx.global_history.items():
        if hist is None or hist.empty or "Close" not in hist:
            continue
        series = hist["Close"]
        name = names.get(ticker, ticker)
        if ticker == "^VIX":
            level = float(series.iloc[-1])
            score = ta.clamp((20.0 - level) / 12.0)
            value = f"{level:.1f} ({'calm' if level < 18 else 'elevated' if level > 24 else 'neutral'})"
        else:
            ret = ta.pct_return(series, min(21, len(series) - 1))
            if ret is None:
                continue
            score = ta.clamp(ret / 0.06)
            value = f"{ret * 100:+.1f}% over 1M"
        global_parts.append(score)
        signals.append(Signal(name=f"Global: {name}", value=value, impact=score))
    if global_parts:
        sub["global"] = float(np.mean(global_parts))

    # Trend --------------------------------------------------------------------
    votes = []
    for n in (20, 50, 200):
        ma = lv[f"sma{n}"]
        if ma is None:
            continue
        above = close > ma
        votes.append(1.0 if above else -1.0)
        signals.append(Signal(name=f"{n}-DMA", value=f"{'above' if above else 'below'} {m(ma)}",
                              impact=0.3 if above else -0.3))
    if lv["sma50"] and lv["sma200"]:
        golden = lv["sma50"] > lv["sma200"]
        votes.append(1.0 if golden else -1.0)
        signals.append(Signal(name="50/200-DMA", value="golden cross zone" if golden else "death cross zone",
                              impact=0.4 if golden else -0.4))
    sub["trend"] = float(np.mean(votes)) if votes else 0.0

    # Momentum -----------------------------------------------------------------
    mom = []
    rsi = lv["rsi"]
    if rsi is not None:
        if rsi > 75:
            rs_score, note = 0.2, "overbought - stretched"
        elif rsi < 25:
            rs_score, note = -0.6, "oversold - weak"
        else:
            rs_score, note = ta.clamp((rsi - 50) / 25), ""
        mom.append(rs_score)
        signals.append(Signal(name="RSI(14)", value=f"{rsi:.0f}", impact=rs_score, note=note))
    md = ta.macd(c).dropna()
    if len(md) >= 2:
        h_now, h_prev = md["hist"].iloc[-1], md["hist"].iloc[-2]
        macd_score = (0.5 if h_now > 0 else -0.5) + (0.5 if h_now > h_prev else -0.5)
        mom.append(macd_score)
        signals.append(Signal(name="MACD histogram",
                              value=f"{'positive' if h_now > 0 else 'negative'}, {'rising' if h_now > h_prev else 'falling'}",
                              impact=ta.clamp(macd_score)))
    sub["momentum"] = float(np.mean(mom)) if mom else 0.0

    # Relative strength vs index ----------------------------------------------
    rs_parts = []
    idx = ctx.index_history["Close"] if ctx.index_history is not None and not ctx.index_history.empty else None
    for days, label in ((21, "1M"), (63, "3M")):
        r_s = ta.pct_return(c, days)
        r_i = ta.pct_return(idx, days) if idx is not None else None
        if r_s is None or r_i is None:
            continue
        diff = r_s - r_i
        score = ta.clamp(diff / 0.08)
        rs_parts.append(score)
        signals.append(Signal(name=f"vs {ctx.market.index_name} ({label})", value=f"{diff * 100:+.1f} pts",
                              impact=score, note=f"stock {r_s * 100:+.1f}%, index {r_i * 100:+.1f}%"))
    sub["relative_strength"] = float(np.mean(rs_parts)) if rs_parts else 0.0

    # Volume -------------------------------------------------------------------
    vr = ta.volume_ratio(v)
    day_ret = float(c.iloc[-1] / c.iloc[-2] - 1)
    if vr is not None:
        if vr >= 1.5:
            sub["volume"] = 0.7 if day_ret > 0 else -0.7
            label = "accumulation" if day_ret > 0 else "distribution"
        else:
            sub["volume"] = 0.0
            label = "normal"
        signals.append(Signal(name="Volume vs 20D avg", value=f"{vr:.1f}x", impact=sub["volume"], note=label))

    # 52-week range ------------------------------------------------------------
    hi52 = float(c.tail(252).max())
    lo52 = float(c.tail(252).min())
    dist_hi = close / hi52 - 1
    sub["range"] = 0.5 if dist_hi > -0.05 else (-0.4 if dist_hi < -0.30 else 0.0)
    signals.append(Signal(name="52-week high", value=f"{dist_hi * 100:+.1f}% from {m(hi52)}", impact=sub["range"],
                          note=f"52-week low {m(lo52)}"))

    used = {k: w for k, w in WEIGHTS.items() if k in sub}
    score = sum(sub[k] * w for k, w in used.items()) / sum(used.values())
    coverage = 1.0 if len(df) >= 200 else 0.75 if len(df) >= 60 else 0.45
    agreement = 1 - min(0.5, float(np.std(list(sub.values()))) / 2)
    confidence = round(coverage * agreement, 3)

    trend_word = "an uptrend" if sub["trend"] > 0.3 else "a downtrend" if sub["trend"] < -0.3 else "a sideways trend"
    rationale = (f"{ctx.resolved.symbol} is in {trend_word} at {m(close)} ({day_ret * 100:+.2f}% on the day). "
                 f"Momentum reads {'positive' if sub['momentum'] > 0 else 'negative'}"
                 + (f" with RSI at {rsi:.0f}" if rsi is not None else "")
                 + (f"; relative strength vs {ctx.market.index_name} is "
                    f"{'leading' if sub['relative_strength'] > 0 else 'lagging'}." if rs_parts else ".")
                 + (f" Global risk tone is {'supportive' if sub['global'] > 0.1 else 'cautious' if sub['global'] < -0.1 else 'neutral'}."
                    if "global" in sub else ""))

    signals.sort(key=lambda s: abs(s.impact), reverse=True)
    return AgentReport(agent="market", score=ta.clamp(score), confidence=confidence, signals=signals,
                       rationale=rationale, data_ok=True)

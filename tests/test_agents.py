import numpy as np
import pytest

from stockdesk.agents import chief_analyst, fundamentals_agent, market_agent, news_agent, sentiment_agent
from stockdesk.agents.context import StockContext, gather
from stockdesk.models import AgentReport, Verdict


def ctx_for(demo, symbol, horizon="positional"):
    return gather(demo, demo.resolve(symbol), horizon)


def test_gather_collects_everything(demo):
    ctx = ctx_for(demo, "KESTREL")
    assert len(ctx.history) >= 200 and not ctx.index_history.empty
    assert ctx.fundamentals["roe"] is not None
    assert ctx.news and not ctx.errors


def test_market_agent_separates_trends(demo):
    up = market_agent.run(ctx_for(demo, "KESTREL"))
    down = market_agent.run(ctx_for(demo, "MONSOON"))
    assert up.score > 0.2 > down.score
    assert all(-1 <= s.impact <= 1 for s in up.signals)


def test_market_agent_includes_global_backdrop(demo):
    ctx = ctx_for(demo, "KESTREL")
    ctx.market = type(ctx.market)(**{**ctx.market.__dict__, "global_benchmarks": (("WORLD", "World index"),)})
    ctx.global_history = {"WORLD": demo.history(demo.info.index_ticker)}
    report = market_agent.run(ctx)
    assert any(signal.name == "Global: World index" for signal in report.signals)
    assert "Global risk tone" in report.rationale


def test_market_agent_handles_missing_history(demo):
    ctx = ctx_for(demo, "KESTREL")
    ctx.history = ctx.history.head(10)
    rep = market_agent.run(ctx)
    assert rep.data_ok is False and rep.confidence == 0


def test_sentiment_and_news_agree_on_bad_news(demo):
    ctx = ctx_for(demo, "WAVELINE")
    sent = sentiment_agent.run(ctx)
    news = news_agent.run(ctx)
    assert sent.score < 0 and news.score < 0
    assert any(s.name == "Regulatory/legal" for s in news.signals)


@pytest.mark.parametrize("title,event,sign", [
    ("Acme bags Rs 500 crore order from NHAI", "order_win", 1),
    ("Brokerage downgrades Acme, cuts target", "rating_change", -1),
    ("SEBI issues show cause notice to Acme", "regulatory_legal", -1),
    ("Acme CFO resigns with immediate effect", "management", -1),
    ("Acme Q2 net profit jumps 28%", "earnings", 1),
])
def test_news_rules(title, event, sign):
    ev, impact, materiality = news_agent.classify_rules(title)
    assert ev == event and np.sign(impact) == sign and materiality >= 0.5


def test_news_agent_uses_llm_when_available(demo, monkeypatch):
    ctx = ctx_for(demo, "NILGIRI")  # routine headlines -> rules see nothing material

    class FakeLLM:
        available = True
        label = "fake"
        def complete_json(self, system, user, schema):
            return schema(events=[{"index": 0, "event_type": "guidance", "impact": 0.6,
                                   "materiality": 0.8, "why": "test"}])
    monkeypatch.setattr(news_agent, "get_llm", lambda: FakeLLM())
    rep = news_agent.run(ctx)
    assert rep.score == pytest.approx(0.6) and "LLM" in rep.rationale


def test_fundamentals_quality(demo):
    good = fundamentals_agent.run(ctx_for(demo, "KESTREL"))
    bad = fundamentals_agent.run(ctx_for(demo, "MONSOON"))
    assert good.score > 0 > bad.score
    assert good.confidence > 0.5


def test_fundamentals_skips_leverage_for_banks(demo):
    rep = fundamentals_agent.run(ctx_for(demo, "ZENITH"))
    assert not any(s.name == "Debt / equity" for s in rep.signals)


# --- Chief analyst rules -------------------------------------------------------------

def _rep(agent, score, conf=0.8, ok=True):
    return AgentReport(agent=agent, score=score, confidence=conf, data_ok=ok, rationale=f"{agent} says {score}.")


def _reports(**scores):
    return {a: _rep(a, *(v if isinstance(v, tuple) else (v,))) for a, v in scores.items()}


def test_chief_buy_and_sell(demo):
    ctx = ctx_for(demo, "BANYAN")
    buy = chief_analyst.decide(ctx, _reports(market=0.6, sentiment=0.5, news=0.4, fundamentals=0.5))
    sell = chief_analyst.decide(ctx, _reports(market=-0.6, sentiment=-0.5, news=-0.3, fundamentals=-0.5))
    assert buy.verdict is Verdict.BUY
    assert sell.verdict is Verdict.SELL
    assert buy.flips_if and sell.flips_if
    assert buy.investor_call.posture == "ACCUMULATE"
    assert buy.investor_call.buy_zone_low < buy.investor_call.buy_zone_high
    assert buy.investor_call.invalidation_price
    assert sell.investor_call.posture == "AVOID" and sell.investor_call.buy_zone_low is None
    assert sell.investor_call.trigger_price


def test_chief_conflict_forces_hold(demo):
    rec = chief_analyst.decide(ctx_for(demo, "BANYAN"),
                               _reports(market=0.7, sentiment=0.2, news=-0.6, fundamentals=0.6))
    assert rec.verdict is Verdict.HOLD
    assert any("Conflicting" in f for f in rec.rule_flags)
    assert rec.investor_call.posture == "WAIT"
    assert rec.investor_call.trigger_price or rec.investor_call.buy_zone_low


def test_chief_no_price_data_forces_hold(demo):
    rec = chief_analyst.decide(ctx_for(demo, "BANYAN"),
                               _reports(market=(0.0, 0.0, False), sentiment=0.6, news=0.6, fundamentals=0.6))
    assert rec.verdict is Verdict.HOLD


def test_chief_overextended_turns_buy_into_hold(demo):
    ctx = ctx_for(demo, "BANYAN")
    close = ctx.history["Close"].to_numpy().copy()
    close[-15:] = close[-16] * np.cumprod(np.full(15, 1.03))   # 15 straight up days -> RSI ~ 100
    ctx.history = ctx.history.assign(Close=close)
    rec = chief_analyst.decide(ctx, _reports(market=0.7, sentiment=0.6, news=0.5, fundamentals=0.6))
    assert rec.verdict is Verdict.HOLD
    assert any("Overextended" in f for f in rec.rule_flags)
    assert rec.investor_call.buy_zone_low < rec.investor_call.buy_zone_high


def test_chief_low_confidence_is_hold(demo):
    rec = chief_analyst.decide(ctx_for(demo, "BANYAN"),
                               _reports(market=(0.9, 0.2), sentiment=(0.9, 0.1), news=(0.9, 0.1), fundamentals=(0.9, 0.2)))
    assert rec.verdict is Verdict.HOLD and any("Low overall confidence" in f for f in rec.rule_flags)


def test_horizon_changes_weighting(demo):
    reports = _reports(market=0.8, sentiment=0.0, news=0.0, fundamentals=-0.6)
    swing, _ = chief_analyst.combine(reports, "swing")
    long, _ = chief_analyst.combine(reports, "long")
    assert swing > long

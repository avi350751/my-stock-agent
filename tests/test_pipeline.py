import time
from pathlib import Path

import pytest

from stockdesk import ledger
from stockdesk.graph import analyze, build_graph
from stockdesk.markets.registry import get_market
from stockdesk.models import AgentReport, Verdict
from stockdesk.report.deliver import save_analysis_report
from stockdesk.flows.check import run_check
from stockdesk.web.server import result_payload


def test_graph_runs_all_four_agents(demo):
    rep, trace = analyze("kestrel", adapter=demo)
    assert set(rep.reports) == {"market", "sentiment", "news", "fundamentals"}
    assert rep.snapshot.symbol == "KESTREL" and rep.chart_png and Path(rep.chart_png).exists()
    assert trace[0].startswith("resolve") and trace[-1].startswith("chief")


def test_analysts_run_in_parallel(demo, monkeypatch):
    """Four analysts sleeping 0.4s each must finish in well under 4 x 0.4s."""
    import stockdesk.graph as g

    def slow(name):
        def fn(ctx):
            time.sleep(0.4)
            return AgentReport(agent=name, score=0.1, confidence=0.5, rationale=f"{name} ok.")
        return fn

    for mod in ("market_agent", "sentiment_agent", "news_agent", "fundamentals_agent"):
        monkeypatch.setattr(getattr(g, mod), "run", slow(mod.split("_")[0]))
    graph = build_graph()
    t0 = time.perf_counter()
    state = graph.invoke({"query": "BANYAN", "horizon": "positional", "narrate": False},
                         config={"configurable": {"adapter": demo}})
    elapsed = time.perf_counter() - t0
    assert set(state["reports"]) == {"market", "sentiment", "news", "fundamentals"}
    assert elapsed < 1.2, f"analysts look sequential ({elapsed:.2f}s)"


def test_failing_agent_degrades_gracefully(demo, monkeypatch):
    import stockdesk.graph as g
    monkeypatch.setattr(g.news_agent, "run", lambda ctx: 1 / 0)
    rep, _ = analyze("BANYAN", adapter=demo, chart=False)
    assert rep.reports["news"].data_ok is False and rep.recommendation.verdict in Verdict


def test_single_stock_html_report_is_local_only(demo):
    rep, _ = analyze("ZENITH", adapter=demo)
    path = save_analysis_report(rep, demo.info)
    html = path.read_text()
    assert "Stock analysis" in html and "data:image/png;base64" in html
    assert "Investor call" in html and "Indicative buy zone" in html


def test_web_payload_contains_investor_call(demo):
    payload = result_payload(run_check("KESTREL", adapter=demo, narrate=False), demo)
    call = payload["investor_call"]
    assert call["posture"] == "ACCUMULATE"
    assert call["buy_zone"] and call["invalidation"]


def test_ledger_roundtrip(demo):
    rep, _ = analyze("CEDAR", adapter=demo, chart=False)
    ledger.record(rep, "analyze")
    ledger.record(rep, "analyze")   # idempotent per day/mode/ticker
    df = ledger.load()
    assert len(df) == 1 and df.iloc[0]["symbol"] == "CEDAR"


def test_market_registry_has_us_provision():
    assert get_market("NSE").info.currency == "INR"
    us = get_market("US")
    assert us.info.currency_symbol == "$"
    with pytest.raises(NotImplementedError):
        us.resolve("AAPL")
    with pytest.raises(KeyError):
        get_market("LSE")

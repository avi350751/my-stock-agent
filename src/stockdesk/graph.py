"""LangGraph orchestration.

    resolve -> gather -> ┬─ market ───────┐
                         ├─ sentiment ────┤
                         ├─ news ─────────┼─> chief -> END
                         └─ fundamentals ─┘

The four analyst nodes run in the same super-step, i.e. in parallel. Their outputs are
merged into `reports` by a dict reducer; `chief` runs once all four have finished.
"""
from __future__ import annotations

import logging
import operator
from types import ModuleType
from typing import Annotated, Any, Callable, TypedDict

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from stockdesk.agents import (chief_analyst, fundamentals_agent, market_agent, news_agent,
                              sentiment_agent)
from stockdesk.agents.context import StockContext, gather
from stockdesk.markets.base import MarketAdapter, ResolvedSymbol
from stockdesk.markets.registry import get_market
from stockdesk.models import AgentReport, StockReport, StockSnapshot

log = logging.getLogger(__name__)


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class DeskState(TypedDict, total=False):
    query: str
    horizon: str
    narrate: bool
    use_openai: bool
    resolved: ResolvedSymbol
    prefetched_history: pd.DataFrame
    prefetched_index: pd.DataFrame
    ctx: StockContext
    reports: Annotated[dict[str, AgentReport], _merge]
    trace: Annotated[list[str], operator.add]
    result: StockReport


def _adapter(config: RunnableConfig | None) -> MarketAdapter:
    cfg = (config or {}).get("configurable", {})
    return cfg.get("adapter") or get_market(cfg.get("market"))


def resolve_node(state: DeskState, config: RunnableConfig) -> dict:
    if state.get("resolved"):
        return {"trace": ["resolve: pre-resolved"]}
    resolved = _adapter(config).resolve(state["query"])
    return {"resolved": resolved, "trace": [f"resolve: {state['query']} -> {resolved.ticker}"]}


def gather_node(state: DeskState, config: RunnableConfig) -> dict:
    ctx = gather(_adapter(config), state["resolved"], state.get("horizon", "positional"),
                 history=state.get("prefetched_history"), index_history=state.get("prefetched_index"),
                 use_openai=state.get("use_openai", state.get("narrate", True)))
    return {"ctx": ctx, "trace": [f"gather: {len(ctx.history)} bars, {len(ctx.news)} news, "
                                  f"{len(ctx.peer_fundamentals)} peers, errors={ctx.errors or 'none'}"]}


def _analyst(name: str, module: ModuleType) -> Callable[[DeskState], dict]:
    def node(state: DeskState) -> dict:
        try:
            report = module.run(state["ctx"])  # looked up at call time (easy to swap/mocks)
        except Exception as exc:  # an analyst crash degrades confidence, it never sinks the run
            log.exception("%s agent failed", name)
            report = AgentReport(agent=name, score=0.0, confidence=0.0, data_ok=False,
                                 rationale=f"{name} analysis failed.", warnings=[repr(exc)])
        return {"reports": {name: report}, "trace": [f"{name}: score {report.score:+.2f} conf {report.confidence:.2f}"]}
    node.__name__ = f"{name}_node"
    return node


def chief_node(state: DeskState) -> dict:
    ctx, reports = state["ctx"], state["reports"]
    rec = chief_analyst.decide(ctx, reports)
    if state.get("narrate", True):
        rec.note = chief_analyst.narrate(ctx, reports, rec)

    h = ctx.history
    last, prev = (float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])) if len(h) >= 2 else (float("nan"),) * 2
    snap = StockSnapshot(
        symbol=ctx.resolved.symbol, ticker=ctx.resolved.ticker, name=ctx.name, market=ctx.market.key,
        currency=ctx.market.currency, sector=ctx.fundamentals.get("sector", ""),
        industry=ctx.resolved.industry or ctx.fundamentals.get("industry", ""),
        last_close=last, prev_close=prev, change_pct=(last / prev - 1) * 100 if prev else 0.0,
        as_of=h.index[-1].date() if len(h) else pd.Timestamp.today().date(),
    )
    result = StockReport(snapshot=snap, horizon=ctx.horizon, reports=reports, recommendation=rec)
    return {"result": result, "trace": [f"chief: {rec.verdict.value} composite {rec.composite:+.2f}"]}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("resolve", resolve_node)
    g.add_node("gather", gather_node)
    analysts = {
        "market": market_agent,
        "sentiment": sentiment_agent,
        "news": news_agent,
        "fundamentals": fundamentals_agent,
    }
    for name, module in analysts.items():
        g.add_node(name, _analyst(name, module))
    g.add_node("chief", chief_node)

    g.add_edge(START, "resolve")
    g.add_edge("resolve", "gather")
    for name in analysts:
        g.add_edge("gather", name)
    g.add_edge(list(analysts), "chief")
    g.add_edge("chief", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def attach_chart(report: StockReport, ctx: StockContext) -> StockReport:
    """Render the price chart used by the web UI and local HTML report."""
    from stockdesk.config import get_settings
    from stockdesk.report.chart import price_chart
    if len(ctx.history) >= 20:
        path = get_settings().out_dir / f"chart_{report.snapshot.symbol.replace('&', '_')}_{report.snapshot.as_of}.png"
        report.chart_png = str(price_chart(ctx.history, path, currency_symbol=ctx.market.currency_symbol))
    return report


def analyze(query: str, horizon: str = "positional", adapter: MarketAdapter | None = None,
            narrate: bool = True, chart: bool = True, **extra: Any) -> tuple[StockReport, list[str]]:
    """Run the full graph for one stock. Returns the report and the execution trace."""
    state = get_graph().invoke({"query": query, "horizon": horizon, "narrate": narrate,
                                "use_openai": narrate, **extra},
                               config={"configurable": {"adapter": adapter} if adapter else {}})
    report = state["result"]
    if chart:
        attach_chart(report, state["ctx"])
    return report, state.get("trace", [])

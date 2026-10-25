"""One stock name or ticker in; one BUY, HOLD or SELL decision out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from stockdesk.config import get_settings
from stockdesk.graph import attach_chart, get_graph
from stockdesk.markets.base import MarketAdapter, ResolvedSymbol
from stockdesk.markets.registry import get_market
from stockdesk.models import StockReport

ProgressFn = Callable[[str, str], None]   # (stage, message)


@dataclass
class CheckResult:
    report: StockReport
    trace: list[str] = field(default_factory=list)


def resolve(query: str, adapter: MarketAdapter | None = None) -> ResolvedSymbol:
    """Name or ticker -> symbol. Raises AmbiguousSymbol (with candidates) or SymbolNotFound."""
    return (adapter or get_market()).resolve(query)


def run_check(stock: str | ResolvedSymbol, horizon: str | None = None, adapter: MarketAdapter | None = None,
              progress: ProgressFn | None = None, chart: bool = False, narrate: bool = True) -> CheckResult:
    """Resolve, fan out to the four analysts in parallel, and return the call.

    `progress` is called as each graph node finishes, so a UI can show the agents
    completing in real time (they finish in whatever order the data allows).
    """
    adapter = adapter or get_market()
    horizon = horizon or get_settings().default_horizon
    notify = progress or (lambda stage, msg: None)

    resolved = stock if isinstance(stock, ResolvedSymbol) else resolve(stock, adapter)
    notify("resolve", f"{resolved.name} ({resolved.symbol}) → {resolved.ticker}")

    state: dict = {}
    trace: list[str] = []
    inputs = {"query": resolved.symbol, "resolved": resolved, "horizon": horizon,
              "narrate": narrate, "use_openai": narrate}
    for update in get_graph().stream(inputs, config={"configurable": {"adapter": adapter}}, stream_mode="updates"):
        for node, delta in update.items():
            if not delta:
                continue
            state.update({k: v for k, v in delta.items() if k not in ("reports", "trace")})
            trace += delta.get("trace", [])
            if node == "gather":
                ctx = delta["ctx"]
                notify("gather", f"{len(ctx.history)} price sessions, {len(ctx.global_history)} global benchmarks, "
                                 f"{len(ctx.news)} headlines, {len(ctx.peer_fundamentals)} industry peers")
            elif node in ("market", "sentiment", "news", "fundamentals"):
                rep = delta["reports"][node]
                notify(node, f"{rep.score:+.2f} (confidence {rep.confidence:.0%})")
            elif node == "chief":
                notify("chief", delta["result"].recommendation.verdict.value)

    report: StockReport = state["result"]
    if chart:
        attach_chart(report, state["ctx"])
    return CheckResult(report=report, trace=trace)

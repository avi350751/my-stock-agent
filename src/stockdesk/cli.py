"""stockdesk command line: one stock in, one BUY / HOLD / SELL decision out."""
from __future__ import annotations

import logging
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stockdesk import ledger
from stockdesk.config import get_settings
from stockdesk.markets.base import AmbiguousSymbol, MarketAdapter, SymbolNotFound
from stockdesk.markets.registry import available, get_market

app = typer.Typer(add_completion=False, help="Open-source multi-agent stock research desk. "
                  "Run with no command to check stocks interactively.")
console = Console()

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context,
         verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "matplotlib", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.DEBUG if verbose else logging.CRITICAL)
    if ctx.invoked_subcommand is None:
        interactive()


_ACTION_COLOR = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}
_AGENT_LABEL = {"market": "Market analyst", "sentiment": "Sentiment analyst", "news": "News analyst",
                "fundamentals": "Fundamentals analyst"}


def _progress(stage: str, msg: str) -> None:
    if stage == "resolve":
        console.print(f"[green]✓[/] Found      {msg}")
    elif stage == "gather":
        console.print(f"[green]✓[/] Data       {msg}")
    elif stage in _AGENT_LABEL:
        console.print(f"[green]✓[/] {_AGENT_LABEL[stage]:<21} {msg}")


def _pick_candidate(exc: AmbiguousSymbol):
    """Let the user choose when a name matches several stocks."""
    console.print(f"[yellow]'{exc.query}' matches several stocks:[/]")
    for i, c in enumerate(exc.candidates, 1):
        console.print(f"  {i}. {c.name} ({c.symbol})")
    if not sys.stdin.isatty():
        return None
    choice = typer.prompt("Pick a number (or press Enter to cancel)", default="", show_default=False).strip()
    if choice.isdigit() and 1 <= int(choice) <= len(exc.candidates):
        return exc.candidates[int(choice) - 1]
    return None


def _print_decision(result, adapter: MarketAdapter, details: bool) -> None:
    rep = result.report
    s, rec = rep.snapshot, rep.recommendation
    cur = adapter.info.currency_symbol
    arrow = "▲" if s.change_pct >= 0 else "▼"
    title = (f"[bold]{s.name}[/] ({s.symbol}) · {cur}{s.last_close:,.2f} "
             f"[{'green' if s.change_pct >= 0 else 'red'}]{arrow} {abs(s.change_pct):.2f}%[/] · "
             f"{adapter.info.key} close {s.as_of:%d %b %Y}")
    word = rec.verdict.value
    lines = [f"Decision  →  [bold {_ACTION_COLOR[word]}]{word}[/]"]
    border = _ACTION_COLOR[word]
    lines += ["", f"[dim]Signal {rec.composite:+.2f} · confidence {rec.confidence:.0%} · "
                  f"{rep.horizon} horizon[/]", "", rec.note]
    console.print()
    console.print(Panel("\n".join(lines), title=title, border_style=border, padding=(1, 2)))

    plan = rec.investor_call
    plan_lines = [f"[bold]Posture[/]  {plan.posture}"]
    if plan.buy_zone_low and plan.buy_zone_high:
        plan_lines.append(f"[bold]Indicative buy zone[/]  {cur}{plan.buy_zone_low:,.2f} – {cur}{plan.buy_zone_high:,.2f}")
    if plan.trigger_price:
        plan_lines.append(f"[bold]Entry trigger[/]  {cur}{plan.trigger_price:,.2f}")
    if plan.invalidation_price:
        plan_lines.append(f"[bold]Thesis invalid below[/]  {cur}{plan.invalidation_price:,.2f}")
    plan_lines += [plan.guidance, f"[dim]{plan.basis} Levels are indicative as of {s.as_of:%d %b %Y}.[/]"]
    console.print(Panel("\n".join(plan_lines), title="Investor call", border_style="cyan", padding=(1, 2)))

    signals = sorted((sg for r in rep.reports.values() for sg in r.signals), key=lambda x: x.impact, reverse=True)
    pros = [sg for sg in signals if sg.impact > 0.15][:3]
    cons = [sg for sg in reversed(signals) if sg.impact < -0.15][:3]
    if pros:
        console.print("[bold green]Working for it[/]")
        for sg in pros:
            console.print(f"  + {sg.name}: {sg.value}")
    if cons:
        console.print("[bold red]Working against it[/]")
        for sg in cons:
            console.print(f"  − {sg.name}: {sg.value}")
    if rec.flips_if:
        console.print("[bold]What would change the call[/]")
        for f in rec.flips_if:
            console.print(f"  • {f}")
    for f in rec.rule_flags:
        console.print(f"[yellow]  ! {f}[/]")

    if details:
        t = Table(title="Analyst reports (ran in parallel)", expand=True)
        t.add_column("Agent", style="bold")
        t.add_column("Score", justify="right")
        t.add_column("Conf", justify="right")
        t.add_column("Top signals")
        for key in ("market", "sentiment", "news", "fundamentals"):
            r = rep.reports.get(key)
            if not r:
                continue
            sc = "green" if r.score >= 0.15 else "red" if r.score <= -0.15 else "yellow"
            sig = "; ".join(f"{x.name}: {x.value}" for x in r.signals[:3]) or r.rationale
            t.add_row(key, f"[{sc}]{r.score:+.2f}[/]", f"{r.confidence:.0%}", sig)
        console.print(t)
        for key in ("news", "sentiment"):
            r = rep.reports.get(key)
            for ev in (r.evidence[:3] if r else []):
                if ev.url:
                    console.print(f"  [dim]{ev.source or key}: {ev.title}[/]\n  [dim]{ev.url}[/]")


def _run_check(stock: str, horizon: str | None, details: bool, report: bool,
               adapter: MarketAdapter | None = None, sample: bool = False) -> bool:
    """Shared by `check`, interactive mode and `demo --stock`. Returns False if nothing was analysed."""
    from stockdesk.flows.check import resolve, run_check
    from stockdesk.report.deliver import save_analysis_report
    adapter = adapter or get_market()
    try:
        target = resolve(stock, adapter)
    except AmbiguousSymbol as exc:
        target = _pick_candidate(exc)
        if target is None:
            return False
    except SymbolNotFound as exc:
        console.print(f"[red]{exc}[/]")
        return False

    console.print(f"\nChecking [bold]{stock}[/] - four analysts working in parallel …")
    try:
        with console.status("Analysing …"):
            result = run_check(target, horizon=horizon, adapter=adapter, progress=_progress, chart=report,
                               narrate=not sample)
    except Exception as exc:  # data source down etc. - keep the interactive loop alive
        console.print(f"[red]Analysis failed: {exc}[/]  (run `stockdesk doctor` to check data sources)")
        return False
    _print_decision(result, adapter, details)
    if not sample:
        ledger.record(result.report, mode="check")
    if report:
        path = save_analysis_report(result.report, adapter.info, sample=sample)
        console.print(f"\nHTML report saved: {path}")
    return True


@app.command()
def check(
    stock: str = typer.Argument(..., help="Stock name or NSE ticker, e.g. INFY, RELIANCE, 'tata motors', 'hdfc bank'"),
    horizon: str = typer.Option(None, "--horizon", "-h", help="swing | positional | long"),
    details: bool = typer.Option(False, "--details", "-d", help="Show every analyst's report and news links"),
    report: bool = typer.Option(False, "--report", help="Also save a local HTML report"),
) -> None:
    """Check one stock and return BUY, HOLD or SELL."""
    if not _run_check(stock, horizon, details, report):
        raise typer.Exit(2)


@app.command("analyze", hidden=True)
def analyze_alias(
    stock: str = typer.Argument(...),
    horizon: str = typer.Option(None, "--horizon", "-h"),
    details: bool = typer.Option(False, "--details", "-d"),
    report: bool = typer.Option(False, "--report"),
) -> None:
    """Alias of `check`."""
    check(stock, horizon, details, report)


def interactive() -> None:
    """Type a stock name or ticker, get the call. Blank line or 'q' to quit."""
    s = get_settings()
    console.print(Panel.fit("[bold]stockdesk[/] · single-stock check\n"
                            "Type a stock name or ticker (e.g. [bold]INFY[/], [bold]tata motors[/], "
                            "[bold]hdfc bank[/]).\nAdd [bold]-d[/] for analyst details. "
                            "Press Enter on an empty line to quit.",
                            border_style="green"))
    while True:
        try:
            raw = console.input("\n[bold green]Stock › [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("", "q", "quit", "exit"):
            break
        tokens = raw.split()
        details = "-d" in tokens or "--details" in tokens
        stock = " ".join(tk for tk in tokens if not tk.startswith("-"))
        if stock:
            _run_check(stock, s.default_horizon, details, report=False)
    console.print("[dim]Bye.[/]")


@app.command()
def demo(
    stock: str = typer.Option(None, "--stock", help="Run the single-stock check on a demo company"),
) -> None:
    """Offline preview with fictional companies and synthetic data - no network needed."""
    from stockdesk.demo import DemoAdapter
    adapter = DemoAdapter()
    _run_check(stock or "KESTREL", get_settings().default_horizon, details=False, report=True,
               adapter=adapter, sample=True)


@app.command()
def serve(
    port: int = typer.Option(8765, "--port", "-p", help="Port on localhost"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (keep localhost unless you know why)"),
    demo_mode: bool = typer.Option(False, "--demo", help="Use fictional demo companies (no network needed)"),
    open_browser: bool = typer.Option(False, "--open", help="Open the page in your default browser"),
) -> None:
    """Web page for the single-stock check: type a name or ticker, get BUY / HOLD / SELL."""
    from stockdesk.web.server import make_server
    if demo_mode:
        from stockdesk.demo import DemoAdapter
        adapter = DemoAdapter()
    else:
        adapter = get_market()
    server = make_server(adapter, get_settings().default_horizon, host=host, port=port, demo=demo_mode)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    console.print(Panel.fit(f"[bold]stockdesk[/] single-stock check is running at [bold green]{url}[/]\n"
                            f"{'[yellow]Demo data[/] · ' if demo_mode else ''}Press Ctrl+C to stop.", border_style="green"))
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        console.print("[dim]Stopped.[/]")


@app.command()
def doctor() -> None:
    """Check data sources, the LLM and sentiment model."""
    import httpx
    from stockdesk.analysis.sentiment import get_scorer
    from stockdesk.llm.client import get_llm
    s = get_settings()
    t = Table(title="stockdesk doctor")
    t.add_column("Check")
    t.add_column("Status")
    t.add_column("Detail")

    def row(name, ok, detail=""):
        t.add_row(name, "[green]OK[/]" if ok else "[red]FAIL[/]", detail)

    adapter = get_market()
    try:
        idx = adapter.history(adapter.info.index_ticker, "5d")
        row("Prices (Yahoo Finance)", not idx.empty, f"last {adapter.info.index_name} bar: {idx.index[-1].date()}" if not idx.empty else "no data")
    except Exception as exc:
        row("Prices (Yahoo Finance)", False, str(exc)[:80])
    uni = adapter.universe(s.universe)
    row("Universe", len(uni) > 50, f"{len(uni)} stocks in {s.universe}" + ("" if len(uni) > 50 else " (bundled fallback)"))
    try:
        r = httpx.get("https://news.google.com/rss/search?q=NSE&hl=en-IN&gl=IN&ceid=IN:en", timeout=10)
        row("News (Google News RSS)", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as exc:
        row("News (Google News RSS)", False, exc.__class__.__name__)
    llm = get_llm()
    row("OpenAI", llm.available, llm.label if llm.available else "set OPENAI_API_KEY; deterministic analysis remains available")
    row("Sentiment", True, get_scorer().name)
    row("Markets", True, f"active {s.market}; registered {', '.join(available())}")
    console.print(t)


@app.command("ledger")
def ledger_cmd(score: bool = typer.Option(False, "--score", help="Score past calls vs the index")) -> None:
    """Show logged recommendations, or score them against the index."""
    if not score:
        df = ledger.load()
        console.print(df.tail(30).to_string(index=False) if not df.empty else "Ledger is empty.")
        return
    scored = ledger.score(get_market())
    if scored.empty:
        console.print("Nothing old enough to score yet.")
        return
    console.print(scored.to_string(index=False))
    console.print("\n[bold]By verdict[/] (excess return vs index, %)")
    console.print(ledger.summary(scored).to_string())


if __name__ == "__main__":
    app()

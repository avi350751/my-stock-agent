"""Tiny local web UI for the single-stock check. Standard library only.

    stockdesk serve            -> http://localhost:8765

Endpoints
  GET /                      the page
  GET /api/universe          [{symbol, name}] for autocomplete
  GET /api/check?q=INFY      Server-Sent Events: progress per agent, then the result
  GET /api/health            {"ok": true, "market": "NSE", "demo": false}

"""
from __future__ import annotations

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from stockdesk import ledger
from stockdesk.flows.check import CheckResult, resolve, run_check
from stockdesk.markets.base import AmbiguousSymbol, MarketAdapter, SymbolNotFound

log = logging.getLogger(__name__)
AGENTS = ("market", "sentiment", "news", "fundamentals")


def result_payload(res: CheckResult, adapter: MarketAdapter) -> dict:
    rep = res.report
    s, rec = rep.snapshot, rep.recommendation
    cur = adapter.info.currency_symbol
    signals = sorted((sg for r in rep.reports.values() for sg in r.signals), key=lambda x: x.impact, reverse=True)
    pros = [f"{sg.name}: {sg.value}" for sg in signals if sg.impact > 0.15][:4]
    cons = [f"{sg.name}: {sg.value}" for sg in reversed(signals) if sg.impact < -0.15][:4]
    sources, seen = [], set()
    for key in ("news", "sentiment"):
        r = rep.reports.get(key)
        for ev in (r.evidence if r else []):
            title = ev.title.rsplit(" [", 1)[0] if ev.title.endswith("]") else ev.title
            if ev.url and title not in seen:
                seen.add(title)
                sources.append({"title": title, "url": ev.url, "source": ev.source})
    chart = None
    if rep.chart_png and Path(rep.chart_png).exists():
        chart = "data:image/png;base64," + base64.b64encode(Path(rep.chart_png).read_bytes()).decode()
    plan = rec.investor_call
    investor_call = {
        "posture": plan.posture,
        "buy_zone": (f"{cur}{plan.buy_zone_low:,.2f} – {cur}{plan.buy_zone_high:,.2f}"
                     if plan.buy_zone_low and plan.buy_zone_high else None),
        "trigger": f"{cur}{plan.trigger_price:,.2f}" if plan.trigger_price else None,
        "invalidation": f"{cur}{plan.invalidation_price:,.2f}" if plan.invalidation_price else None,
        "guidance": plan.guidance, "basis": plan.basis,
    }
    return {
        "symbol": s.symbol, "name": s.name, "industry": s.industry, "market": s.market,
        "close": f"{cur}{s.last_close:,.2f}", "change_pct": round(s.change_pct, 2),
        "as_of": s.as_of.strftime("%d %b %Y"), "horizon": rep.horizon,
        "verdict": rec.verdict.value, "composite": round(rec.composite, 2), "confidence": round(rec.confidence * 100),
        "note": rec.note, "investor_call": investor_call,
        "pros": pros, "cons": cons, "flips": rec.flips_if, "flags": rec.rule_flags,
        "agents": [{"key": k, "score": round(rep.reports[k].score, 2), "confidence": round(rep.reports[k].confidence * 100),
                    "rationale": rep.reports[k].rationale,
                    "signals": [f"{sg.name}: {sg.value}" for sg in rep.reports[k].signals[:4]]}
                   for k in AGENTS if k in rep.reports],
        "sources": sources[:5], "chart": chart,
    }


class _Handler(BaseHTTPRequestHandler):
    adapter: MarketAdapter
    horizon: str
    demo: bool
    page: bytes
    _universe_cache: list | None = None
    _lock = threading.Lock()

    def log_message(self, fmt, *args):  # keep the terminal quiet
        log.debug(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            return self._send(200, self.page, "text/html; charset=utf-8")
        if url.path == "/api/health":
            return self._json({"ok": True, "market": self.adapter.info.key, "demo": self.demo})
        if url.path == "/api/universe":
            return self._json(self._universe())
        if url.path == "/api/check":
            return self._check(qs.get("q", [""])[0].strip(), qs.get("horizon", [self.horizon])[0])
        return self._send(404, b"not found", "text/plain")

    def _universe(self) -> list:
        cls = type(self)
        with cls._lock:
            if cls._universe_cache is None:
                try:
                    names = ("nifty500",) if "nifty500" in self.adapter.info.universes else (None,)
                    uni = self.adapter.universe(*names)
                    cls._universe_cache = [{"symbol": r.symbol, "name": r.name} for r in uni.itertuples()]
                except Exception as exc:
                    log.warning("universe unavailable: %s", exc)
                    cls._universe_cache = []
        return cls._universe_cache

    def _event(self, name: str, data) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _check(self, query: str, horizon: str) -> None:
        if not query:
            return self._json({"error": "Type a stock name or ticker."}, 400)
        if horizon not in ("swing", "positional", "long"):
            horizon = self.horizon
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            try:
                target = resolve(query, self.adapter)
            except AmbiguousSymbol as exc:
                return self._event("choices", {"query": query, "candidates": [
                    {"symbol": c.symbol, "name": c.name} for c in exc.candidates]})
            except SymbolNotFound as exc:
                return self._event("failed", {"message": str(exc)})
            res = run_check(target, horizon=horizon, adapter=self.adapter, chart=True,
                            progress=lambda stage, msg: self._event("progress", {"stage": stage, "message": msg}))
            if not self.demo:
                ledger.record(res.report, mode="check")
            self._event("result", result_payload(res, self.adapter))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.exception("check failed")
            try:
                self._event("failed", {"message": f"Analysis failed: {exc}. Run `stockdesk doctor` to check data sources."})
            except Exception:
                pass


def make_server(adapter: MarketAdapter, horizon: str, host: str = "127.0.0.1", port: int = 8765,
                demo: bool = False) -> ThreadingHTTPServer:
    page = resources.files("stockdesk.web").joinpath("index.html").read_bytes()
    handler = type("Handler", (_Handler,), {"adapter": adapter, "horizon": horizon, "demo": demo, "page": page,
                                            "_universe_cache": None, "_lock": threading.Lock()})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server

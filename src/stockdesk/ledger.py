"""Recommendation ledger: every call is logged so it can be scored against the index later."""
from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from stockdesk.config import get_settings
from stockdesk.markets.base import MarketAdapter
from stockdesk.models import StockReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    as_of TEXT NOT NULL,
    mode TEXT NOT NULL,
    market TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ticker TEXT NOT NULL,
    horizon TEXT NOT NULL,
    verdict TEXT NOT NULL,
    composite REAL,
    confidence REAL,
    close REAL,
    UNIQUE(as_of, mode, ticker)
);
"""


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(get_settings().ledger_path)
    con.execute(_SCHEMA)
    return con


def record(report: StockReport, mode: str) -> None:
    r, s = report.recommendation, report.snapshot
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO recommendations (logged_at, as_of, mode, market, symbol, ticker, horizon, "
            "verdict, composite, confidence, close) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), s.as_of.isoformat(), mode, s.market, s.symbol, s.ticker,
             report.horizon, r.verdict.value, r.composite, r.confidence, s.last_close))


def load() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql_query("SELECT * FROM recommendations ORDER BY as_of", con, parse_dates=["as_of"])


def score(adapter: MarketAdapter, horizons: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """Forward returns of each logged call vs the index (excess return), in trading days."""
    df = load()
    if df.empty:
        return df
    idx = adapter.history(adapter.info.index_ticker, "2y")["Close"]
    rows = []
    for rec in df.itertuples():
        px = adapter.history(rec.ticker, "2y")["Close"]
        start = pd.Timestamp(rec.as_of)
        if start not in px.index or start not in idx.index:
            continue
        p0, i0 = px.index.get_loc(start), idx.index.get_loc(start)
        row = {"as_of": rec.as_of.date(), "symbol": rec.symbol, "mode": rec.mode, "verdict": rec.verdict}
        for h in horizons:
            if p0 + h < len(px) and i0 + h < len(idx):
                stock_ret = px.iloc[p0 + h] / px.iloc[p0] - 1
                index_ret = idx.iloc[i0 + h] / idx.iloc[i0] - 1
                row[f"excess_{h}d_%"] = round((stock_ret - index_ret) * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    cols = [c for c in scored.columns if c.startswith("excess_")]
    agg = scored.groupby("verdict")[cols].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
    names = {"count": "n", "mean": "avg"}
    agg.columns = [f"{c}_{names.get(f, 'hit%')}" for c, f in agg.columns]
    return agg.round(2)

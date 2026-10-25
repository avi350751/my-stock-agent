"""Typed contracts passed between agents. Every agent returns an AgentReport."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

AgentName = Literal["market", "sentiment", "news", "fundamentals"]


class Signal(BaseModel):
    """One observable fact an agent used, with its directional contribution."""
    name: str
    value: str
    impact: float = Field(ge=-1.0, le=1.0, description="-1 bearish … +1 bullish")
    note: str = ""


class Evidence(BaseModel):
    title: str
    url: str = ""
    source: str = ""
    published: datetime | None = None


class AgentReport(BaseModel):
    agent: AgentName
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[Signal] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str = ""
    data_ok: bool = True
    warnings: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def _round_score(cls, v: float) -> float:
        return round(v, 3)


class Verdict(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class InvestorCall(BaseModel):
    """Deterministic entry guidance derived from observable technical levels."""
    posture: Literal["ACCUMULATE", "WAIT", "AVOID"]
    buy_zone_low: float | None = Field(default=None, gt=0)
    buy_zone_high: float | None = Field(default=None, gt=0)
    trigger_price: float | None = Field(default=None, gt=0)
    invalidation_price: float | None = Field(default=None, gt=0)
    guidance: str
    basis: str


class Recommendation(BaseModel):
    """The chief analyst's single, canonical decision."""
    verdict: Verdict
    composite: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    flips_if: list[str] = Field(default_factory=list, description="Conditions that would change the call")
    rule_flags: list[str] = Field(default_factory=list)
    note: str = ""
    investor_call: InvestorCall


class StockSnapshot(BaseModel):
    symbol: str            # exchange symbol, e.g. RELIANCE
    ticker: str            # data-provider ticker, e.g. RELIANCE.NS
    name: str
    market: str
    currency: str
    sector: str = ""
    industry: str = ""
    last_close: float
    prev_close: float
    change_pct: float
    as_of: date


class StockReport(BaseModel):
    snapshot: StockSnapshot
    horizon: str
    reports: dict[str, AgentReport]
    recommendation: Recommendation
    chart_png: str | None = None

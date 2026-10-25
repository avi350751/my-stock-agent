"""Deterministic finance-lexicon sentiment for recent headlines."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

class Scorer(Protocol):
    name: str
    def score(self, texts: list[str]) -> list[float]: ...


# Compact finance lexicon. Phrases are checked before single words.
_POS_PHRASES = [
    "beats estimates", "beat estimates", "above estimates", "record high", "all-time high", "order win",
    "bags order", "wins order", "target raised", "raises target", "upgrade to buy", "upgraded",
    "strong demand", "margin expansion", "profit jumps", "profit rises", "profit surges", "net profit up",
    "revenue up", "dividend", "buyback", "approval", "capacity expansion", "market share gain",
]
_NEG_PHRASES = [
    "misses estimates", "missed estimates", "below estimates", "target cut", "cuts target", "downgrade",
    "downgraded", "profit falls", "profit drops", "profit declines", "net loss", "margin pressure",
    "weak demand", "show cause", "penalty", "probe", "raid", "default", "fraud", "resigns", "pledge",
    "block deal discount", "warning letter", "import alert", "52-week low", "slump",
]
_POS_WORDS = {
    "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "gain", "gains", "rise",
    "rises", "strong", "robust", "growth", "outperform", "bullish", "upbeat", "record", "beat", "beats",
    "expands", "boost", "boosts", "wins", "positive", "upside", "accumulate",
}
_NEG_WORDS = {
    "fall", "falls", "drop", "drops", "plunge", "plunges", "slump", "slumps", "decline", "declines",
    "weak", "loss", "losses", "bearish", "downside", "cut", "cuts", "miss", "misses", "concern",
    "concerns", "risk", "risks", "lawsuit", "sell", "selloff", "crash", "tumble", "tumbles", "negative",
    "underperform", "slowdown", "delay", "delays",
}
_NEGATORS = {"no", "not", "never", "without", "fails", "failed"}


class LexiconScorer:
    name = "Finance lexicon"

    def score(self, texts: list[str]) -> list[float]:
        return [self._one(t) for t in texts]

    @staticmethod
    def _one(text: str) -> float:
        t = text.lower()
        pos = sum(2 for p in _POS_PHRASES if p in t)
        neg = sum(2 for p in _NEG_PHRASES if p in t)
        tokens = re.findall(r"[a-z\-]+", t)
        for i, tok in enumerate(tokens):
            flip = i > 0 and tokens[i - 1] in _NEGATORS
            if tok in _POS_WORDS:
                neg, pos = (neg + 1, pos) if flip else (neg, pos + 1)
            elif tok in _NEG_WORDS:
                pos, neg = (pos + 1, neg) if flip else (pos, neg + 1)
        total = pos + neg
        return 0.0 if total == 0 else round((pos - neg) / (total + 1), 4)


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    return LexiconScorer()

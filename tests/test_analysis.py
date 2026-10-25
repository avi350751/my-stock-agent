import numpy as np
import pandas as pd
import pytest

from stockdesk.analysis import indicators as ta
from stockdesk.analysis.sentiment import LexiconScorer
from stockdesk.data.news import clean_company_name, dedupe, relevant
from stockdesk.markets.base import NewsItem


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)), dtype=float)


def test_rsi_extremes_and_bounds():
    up = _series(np.linspace(100, 200, 60))
    down = _series(np.linspace(200, 100, 60))
    noisy = _series(100 + np.random.default_rng(0).normal(0, 1, 200).cumsum())
    assert ta.rsi(up).iloc[-1] == pytest.approx(100.0)
    assert ta.rsi(down).iloc[-1] < 5
    r = ta.rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_sma_macd_volume_ratio():
    s = _series(range(1, 61))
    assert ta.sma(s, 10).iloc[-1] == pytest.approx(55.5)
    m = ta.macd(s)
    assert {"macd", "signal", "hist"} <= set(m.columns)
    vol = _series([100] * 20 + [250])
    assert ta.volume_ratio(vol) == pytest.approx(2.5)
    assert ta.pct_return(s, 10) == pytest.approx(60 / 50 - 1)


@pytest.mark.parametrize("text,sign", [
    ("Company Q2 net profit jumps 30%, beats estimates", 1),
    ("Brokerage downgrades stock, cuts target price after weak demand", -1),
    ("Board meeting scheduled for next Tuesday", 0),
    ("Management sees no slowdown in orders", 1),
])
def test_lexicon_direction(text, sign):
    score = LexiconScorer().score([text])[0]
    assert np.sign(score) == sign


def test_news_relevance_uses_word_boundaries():
    items = [NewsItem("L&T bags mega order"), NewsItem("Quarterly results season begins"),
             NewsItem("Larsen & Toubro shares rise")]
    kept = relevant(items, "Larsen & Toubro Ltd.", "LT")
    titles = {i.title for i in kept}
    assert "Quarterly results season begins" not in titles   # 'lt' inside 'results' must not match
    assert "Larsen & Toubro shares rise" in titles


def test_dedupe_and_name_cleaning():
    items = [NewsItem("Infosys wins $1.5 bn deal from European client"),
             NewsItem("Infosys wins $1.5bn deal from European client"),
             NewsItem("Infosys declares dividend")]
    assert len(dedupe(items)) == 2
    assert clean_company_name("Reliance Industries Ltd.") == "Reliance Industries"

"""Technical indicators in plain pandas (no TA-Lib / pandas-ta dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def pct_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def volume_ratio(volume: pd.Series, n: int = 20) -> float | None:
    if len(volume) < n + 1:
        return None
    base = volume.iloc[-n - 1:-1].mean()
    return float(volume.iloc[-1] / base) if base else None


def avg_traded_value(df: pd.DataFrame, n: int = 20) -> float:
    tail = df.tail(n)
    return float((tail["Close"] * tail["Volume"]).mean())


def swing_low(close: pd.Series, lookback: int = 20) -> float:
    return float(close.tail(lookback).min())


def swing_high(close: pd.Series, lookback: int = 20) -> float:
    return float(close.tail(lookback).max())


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

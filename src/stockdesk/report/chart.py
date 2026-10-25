"""Small, calm price charts for the web UI and local HTML report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

TEAL = "#0F766E"
TEAL_FILL = "#CCEFE9"
AMBER = "#D97706"
GRID = "#E6EEE8"
INK = "#5B6F69"


def price_chart(history: pd.DataFrame, out: Path, days: int = 90, currency_symbol: str = "₹") -> Path:
    c = history["Close"]
    sma50 = c.rolling(50).mean()
    c, sma50 = c.tail(days), sma50.tail(days)

    fig, ax = plt.subplots(figsize=(5.6, 1.9), dpi=200)
    lo = float(c.min()) * 0.985
    ax.fill_between(c.index, c.values, lo, color=TEAL_FILL, alpha=0.75, linewidth=0)
    ax.plot(c.index, c.values, color=TEAL, linewidth=1.8, solid_capstyle="round")
    if sma50.notna().any():
        ax.plot(sma50.index, sma50.values, color=AMBER, linewidth=1.1, linestyle=(0, (3, 2.5)), alpha=0.9)
    ax.scatter([c.index[-1]], [c.iloc[-1]], s=22, color=TEAL, zorder=5, edgecolors="white", linewidths=1.2)
    ax.annotate(f"{currency_symbol}{c.iloc[-1]:,.0f}", (c.index[-1], c.iloc[-1]), xytext=(-6, 9),
                textcoords="offset points", ha="right", fontsize=7.5, color=TEAL, fontweight="bold")

    ax.set_ylim(lo, float(c.max()) * 1.02)
    ax.set_xlim(c.index[0], c.index[-1])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=6.5, colors=INK, length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    fig.tight_layout(pad=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out

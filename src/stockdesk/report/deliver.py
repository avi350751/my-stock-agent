"""Save a self-contained local HTML analysis report."""
from __future__ import annotations

import base64
import re
from pathlib import Path

from stockdesk.config import get_settings
from stockdesk.markets.base import MarketInfo
from stockdesk.models import StockReport
from stockdesk.report.html_report import render_analysis


def save_analysis_report(rep: StockReport, info: MarketInfo, sample: bool = False) -> Path:
    """Write a self-contained report that can be opened in any browser."""
    chart_src = None
    if rep.chart_png and Path(rep.chart_png).exists():
        data = base64.b64encode(Path(rep.chart_png).read_bytes()).decode()
        chart_src = f"data:image/png;base64,{data}"
    _, html = render_analysis(rep, info, chart_src, sample=sample)
    path = get_settings().out_dir / f"analysis_{rep.snapshot.symbol}_{rep.snapshot.as_of}.html"
    path.write_text(re.sub(r"<div style=\"display:none.*?</div>", "", html, flags=re.DOTALL), encoding="utf-8")
    return path

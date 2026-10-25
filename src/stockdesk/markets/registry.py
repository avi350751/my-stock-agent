"""Market registry. Add a new market by writing an adapter and adding one line here."""
from __future__ import annotations

from typing import Callable

from stockdesk.markets.base import MarketAdapter

_REGISTRY: dict[str, Callable[..., MarketAdapter]] = {}


def register(key: str, factory: Callable[..., MarketAdapter]) -> None:
    _REGISTRY[key.upper()] = factory


def get_market(key: str | None = None) -> MarketAdapter:
    from stockdesk.config import get_settings
    s = get_settings()
    key = (key or s.market).upper()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown market '{key}'. Registered: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key](default_universe=s.universe) if key == "NSE" else _REGISTRY[key]()


def available() -> list[str]:
    return sorted(_REGISTRY)


def _bootstrap() -> None:
    from stockdesk.markets.nse import NSEAdapter
    from stockdesk.markets.us import USAdapter
    register("NSE", NSEAdapter)
    register("US", USAdapter)   # provision: raises NotImplementedError until implemented


_bootstrap()

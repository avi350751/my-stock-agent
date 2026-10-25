"""Tiny TTL disk cache so repeated runs (and the screener's fan-out) don't hammer free data sources."""
from __future__ import annotations

import hashlib
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from stockdesk.config import get_settings


def _path(key: str) -> Path:
    digest = hashlib.sha1(key.encode()).hexdigest()[:20]
    return get_settings().cache_dir / f"{digest}.pkl"


def get(key: str, ttl_hours: float | None = None) -> Any | None:
    ttl = (ttl_hours if ttl_hours is not None else get_settings().cache_ttl_hours) * 3600
    p = _path(key)
    if not p.exists() or (time.time() - p.stat().st_mtime) > ttl:
        return None
    try:
        with p.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def put(key: str, value: Any) -> None:
    try:
        with _path(key).open("wb") as fh:
            pickle.dump(value, fh)
    except Exception:
        pass


def cached(prefix: str, ttl_hours: float | None = None) -> Callable:
    """Cache a method/function result on its positional + keyword args (ignores `self`)."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key_args = args[1:] if args and hasattr(args[0], "__dict__") else args
            key = f"{prefix}:{key_args!r}:{sorted(kwargs.items())!r}"
            hit = get(key, ttl_hours)
            if hit is not None:
                return hit
            value = fn(*args, **kwargs)
            if value is not None and not (hasattr(value, "empty") and value.empty):
                put(key, value)
            return value
        return wrapper
    return deco

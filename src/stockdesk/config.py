"""Environment-driven configuration. Every knob lives in .env (see .env.example)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Horizon = Literal["swing", "positional", "long"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore",
                                      populate_by_name=True)

    # --- Market -------------------------------------------------------------
    market: str = Field("NSE", description="Active market adapter key (NSE today; US later)")
    universe: str = Field("nifty200", description="Screener universe: nifty50 | nifty100 | nifty200")
    default_horizon: Horizon = "positional"

    min_avg_traded_value_cr: float = Field(25.0, description="Liquidity floor: 20-day avg traded value in INR crore")

    # --- OpenAI ---------------------------------------------------------------
    # A legacy API-key alias keeps existing private credentials working during migration.
    openai_model: str = "gpt-5.5"
    openai_api_key: str = Field("", validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY"))
    openai_timeout_s: float = 90.0

    # --- News -----------------------------------------------------------------
    news_lookback_days: int = 3
    news_max_items: int = 25

    # --- Storage --------------------------------------------------------------
    data_dir: Path = Path(".stockdesk")
    cache_ttl_hours: float = 6.0

    @property
    def cache_dir(self) -> Path:
        p = self.data_dir / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def ledger_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "ledger.sqlite"

    @property
    def out_dir(self) -> Path:
        p = self.data_dir / "out"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Horizon-dependent weights for the chief analyst. Sum to 1.0 per horizon.
HORIZON_WEIGHTS: dict[str, dict[str, float]] = {
    "swing":      {"market": 0.40, "news": 0.25, "sentiment": 0.20, "fundamentals": 0.15},
    "positional": {"market": 0.30, "news": 0.20, "sentiment": 0.15, "fundamentals": 0.35},
    "long":       {"market": 0.15, "news": 0.15, "sentiment": 0.10, "fundamentals": 0.60},
}

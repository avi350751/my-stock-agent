"""OpenAI Responses API client used for classification and plain-English notes.

The model never produces a number used by the investment decision. Structured
outputs are validated with Pydantic; failures fall back to deterministic rules.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from stockdesk.config import Settings, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _output_text(payload: dict) -> str | None:
    """Extract output text from the REST representation of an OpenAI response."""
    parts: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    return "".join(parts).strip() or None


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    @property
    def label(self) -> str:
        return f"{self.s.openai_model} via OpenAI" if self.available else "OpenAI not configured"

    @property
    def available(self) -> bool:
        return bool(self.s.openai_api_key.strip())

    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.2) -> str | None:
        """Generate text with the OpenAI Responses API.

        `json_mode` and `temperature` remain in the interface for callers, but
        output shape is enforced by the prompt and Pydantic validation. This
        keeps the request compatible with OpenAI reasoning models that do not
        expose every sampling parameter.
        """
        if not self.available:
            return None
        body = {"model": self.s.openai_model, "instructions": system, "input": user, "store": False}
        headers = {"Authorization": f"Bearer {self.s.openai_api_key}", "Content-Type": "application/json"}
        try:
            response = httpx.post(_OPENAI_RESPONSES_URL, json=body, headers=headers,
                                  timeout=self.s.openai_timeout_s)
            response.raise_for_status()
            return _output_text(response.json())
        except Exception as exc:
            log.warning("OpenAI request failed: %s", exc)
            return None

    def complete_json(self, system: str, user: str, schema: type[T], retries: int = 1) -> T | None:
        prompt = f"{user}\n\nRespond with ONLY a JSON object matching this schema:\n{json.dumps(schema.model_json_schema())}"
        for _ in range(retries + 1):
            raw = self.complete(system, prompt, json_mode=True)
            if raw is None:
                return None
            match = _JSON_BLOCK.search(raw)
            if not match:
                continue
            try:
                return schema.model_validate_json(match.group(0))
            except ValidationError as exc:
                prompt = f"{prompt}\n\nYour previous answer was invalid: {exc.errors()[:3]}. Fix it."
        return None


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def set_llm(client: LLMClient | None) -> None:
    """Test hook."""
    global _client
    _client = client

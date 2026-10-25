import json

import httpx

from stockdesk.agents import chief_analyst
from stockdesk.agents.news_agent import LLMEvents
from stockdesk.config import Settings
from stockdesk.llm import client as llm_client
from stockdesk.llm.client import LLMClient


class Resp:
    def __init__(self, text: str, status: int = 200):
        self._data = {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def openai_settings() -> Settings:
    return Settings(openai_model="gpt-5.5", openai_api_key="test-key")


def fake_openai(monkeypatch, replies):
    calls = []

    def post(url, json=None, headers=None, **kwargs):
        calls.append({"url": url, "body": json, "headers": headers})
        return Resp(replies.pop(0))

    monkeypatch.setattr(httpx, "post", post)
    return calls


def test_openai_responses_json_roundtrip_with_retry(monkeypatch):
    good = json.dumps({"events": [{"index": 0, "event_type": "order_win", "impact": 0.4,
                                    "materiality": 0.7, "why": "large order"}]})
    calls = fake_openai(monkeypatch, ['{"events": [{"index": 0, "event_type": "bogus"}]}', good])
    out = LLMClient(openai_settings()).complete_json("sys", "headlines", LLMEvents)
    assert out.events[0].event_type == "order_win"
    assert len(calls) == 2 and calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["body"]["model"] == "gpt-5.5" and calls[0]["body"]["store"] is False
    assert "previous answer was invalid" in calls[1]["body"]["input"]


def test_openai_without_key_is_unavailable():
    assert LLMClient(Settings(openai_api_key="")).available is False


def test_legacy_api_key_alias_does_not_restore_local_model_settings(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "legacy-key")
    monkeypatch.setenv("LLM_MODEL", "local-model-name")
    settings = Settings()
    assert settings.openai_api_key == "legacy-key"
    assert settings.openai_model == "gpt-5.5"


def test_narrative_uses_openai_and_falls_back(demo, monkeypatch):
    from stockdesk.graph import analyze
    from stockdesk.agents.context import gather

    report, _ = analyze("CEDAR", adapter=demo, chart=False, narrate=False)
    ctx = gather(demo, demo.resolve("CEDAR"), "positional")

    fake_openai(monkeypatch, ["Cedar is trending higher with supportive news and solid fundamentals today."])
    llm_client.set_llm(LLMClient(openai_settings()))
    assert chief_analyst.narrate(ctx, report.reports, report.recommendation).startswith("Cedar is trending")

    fake_openai(monkeypatch, ["ok"])
    llm_client.set_llm(LLMClient(openai_settings()))
    assert chief_analyst.narrate(ctx, report.reports, report.recommendation) == report.recommendation.note

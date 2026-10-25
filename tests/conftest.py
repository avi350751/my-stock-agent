import pytest

from stockdesk.config import get_settings
from stockdesk.llm import client as llm_client


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test gets its own data dir and no external OpenAI calls."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / ".stockdesk"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()
    llm_client.set_llm(None)
    yield
    get_settings.cache_clear()
    llm_client.set_llm(None)


@pytest.fixture
def demo():
    from stockdesk.demo import DemoAdapter
    return DemoAdapter(as_of="2026-10-23")

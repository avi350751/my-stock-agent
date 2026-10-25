"""One stock name or ticker in -> BUY / HOLD / SELL out."""
import pytest

from typer.testing import CliRunner

from stockdesk import cli
from stockdesk.flows.check import run_check
from stockdesk.models import Verdict


@pytest.mark.parametrize("query", ["CEDAR", "cedar power", "Cedar Power Transmission", "cedar.demo"])
def test_name_or_ticker_runs_full_analysis(demo, query):
    seen = []
    res = run_check(query, adapter=demo, progress=lambda stage, msg: seen.append(stage))
    assert res.report.snapshot.symbol == "CEDAR"
    assert {"resolve", "gather", "market", "sentiment", "news", "fundamentals", "chief"} <= set(seen)
    assert seen.index("gather") < min(seen.index(a) for a in ("market", "sentiment", "news", "fundamentals"))
    assert res.report.recommendation.verdict in Verdict


def test_negative_stock_says_sell(demo):
    assert run_check("MONSOON", adapter=demo).report.recommendation.verdict is Verdict.SELL


@pytest.fixture
def cli_demo(demo, monkeypatch):
    monkeypatch.setattr(cli, "get_market", lambda *a, **k: demo)
    import stockdesk.flows.check as chk
    monkeypatch.setattr(chk, "get_market", lambda *a, **k: demo)
    return CliRunner()


def test_cli_check_returns_one_decision(cli_demo):
    out = cli_demo.invoke(cli.app, ["check", "kestrel"], terminal_width=120)
    assert out.exit_code == 0, out.output
    assert "Decision" in out.output and any(word in out.output for word in ("BUY", "HOLD", "SELL"))
    assert "Market analyst" in out.output and "Fundamentals analyst" in out.output


def test_cli_negative_stock(cli_demo):
    out = cli_demo.invoke(cli.app, ["check", "MONSOON"], terminal_width=120)
    assert out.exit_code == 0 and "Decision" in out.output and "SELL" in out.output


def test_cli_unknown_stock(cli_demo):
    out = cli_demo.invoke(cli.app, ["check", "not a real company"])
    assert out.exit_code == 2 and "not in the demo universe" in out.output


def test_interactive_mode(cli_demo):
    out = cli_demo.invoke(cli.app, [], input="kestrel\nmonsoon\n\n", terminal_width=120)
    assert out.exit_code == 0, out.output
    assert out.output.count("Checking") == 2 and "Decision" in out.output and "Bye." in out.output


def test_interactive_survives_bad_input(cli_demo):
    out = cli_demo.invoke(cli.app, [], input="zzzz\nbanyan\nq\n", terminal_width=120)
    assert out.exit_code == 0 and "not in the demo universe" in out.output and "Checking" in out.output

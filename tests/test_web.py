"""Local web UI for the single-stock check (Flow 1)."""
import json
import threading
import urllib.request

import pytest

from stockdesk.web.server import make_server


@pytest.fixture
def server(demo, monkeypatch):
    srv = make_server(demo, "positional", port=0, demo=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode()


def events(body):
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_page_and_health(server):
    status, ctype, body = get(server + "/")
    assert status == 200 and "text/html" in ctype and "Should I buy, hold or sell?" in body
    assert json.loads(get(server + "/api/health")[2]) == {"ok": True, "market": "NSE", "demo": True}
    uni = json.loads(get(server + "/api/universe")[2])
    assert {"symbol": "KESTREL", "name": "Kestrel Motors Ltd."} in uni


def test_check_streams_progress_then_result(server):
    _, ctype, body = get(server + "/api/check?q=kestrel%20motors")
    assert "text/event-stream" in ctype
    ev = events(body)
    stages = [d["stage"] for e, d in ev if e == "progress"]
    assert stages[:2] == ["resolve", "gather"] and set(stages) >= {"market", "sentiment", "news", "fundamentals", "chief"}
    name, res = ev[-1]
    assert name == "result" and res["symbol"] == "KESTREL"
    assert res["verdict"] in ("BUY", "HOLD", "SELL")
    assert res["chart"].startswith("data:image/png;base64,") and len(res["agents"]) == 4


def test_check_negative_stock(server):
    res = events(get(server + "/api/check?q=MONSOON")[2])[-1][1]
    assert res["verdict"] == "SELL"


def test_unknown_and_empty(server):
    assert events(get(server + "/api/check?q=zzzz")[2])[-1][0] == "failed"
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server + "/api/check?q=")
    assert exc.value.code == 400

"""ClaudePlanner against a local mock of the Messages API (no network, no API key needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from laneige_engine.agent import ClaudePlanner, PlanningSession
from laneige_engine.config import EngineConfig
from laneige_engine.sample_data import demo_inputs


def _message(content, stop_reason):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if len(type(self).requests) == 1:
            reply = _message([{"type": "text", "text": "Checking the portfolio."},
                              {"type": "tool_use", "id": "toolu_1", "name": "get_portfolio_summary", "input": {}}],
                             "tool_use")
        else:
            reply = _message([{"type": "text", "text": "Diwali moves to 8-Nov-2026; October is below LY."}],
                             "end_turn")
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def mock_api():
    _Handler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", _Handler.requests
    server.shutdown()


def test_claude_planner_tool_loop(mock_api, monkeypatch):
    url, requests = mock_api
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    session = PlanningSession(demo_inputs(), EngineConfig(horizon=8))
    planner = ClaudePlanner(session, api_key="test-key")
    events = []
    answer = planner.ask("Why is October below last year?", on_event=lambda k, t: events.append((k, t)))

    assert answer == "Diwali moves to 8-Nov-2026; October is below LY."
    assert ("tool", "get_portfolio_summary()") in events
    assert len(requests) == 2
    first = requests[0]
    assert first["path"].startswith("/v1/messages")
    assert first["body"]["model"] == "claude-opus-5"
    assert first["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in first["headers"].get("anthropic-beta", "")
    assert first["body"]["thinking"] == {"type": "adaptive"}
    assert {t["name"] for t in first["body"]["tools"]} >= {"run_forecast", "get_sku_insights", "set_event_dates"}
    # The tool result sent back is the engine's JSON portfolio summary.
    tool_msg = requests[1]["body"]["messages"][-1]
    result = tool_msg["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    payload = result["content"] if isinstance(result["content"], str) else result["content"][0]["text"]
    assert json.loads(payload)["horizon_buckets"] == 8
    assert planner.history[-1] == {"role": "assistant", "content": answer}

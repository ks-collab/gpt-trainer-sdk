"""The model list must reflect GPT-trainer's current state on every lookup.

Long-lived clients (e.g. a module-level GPTTrainer in a web backend) must not
reject a model that GPT-trainer added after the client's first lookup.

Runs against a local HTTP server; no GPT-trainer credentials needed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gpt_trainer_sdk import GPTTrainer


class _FakeGPTTrainer:
    """Serves /agent/models and /agent/model-costs from a mutable dict."""

    def __init__(self, model_costs: dict[str, int]):
        self.model_costs = model_costs
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/v1/agent/models":
                    body = {"models": list(fake.model_costs)}
                elif self.path == "/api/v1/agent/model-costs":
                    body = {"model_costs": fake.model_costs}
                else:
                    self.send_error(404)
                    return
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_server():
    server = _FakeGPTTrainer({"gpt-4o-8k": 1})
    yield server
    server.close()


@pytest.fixture
def client(fake_server):
    return GPTTrainer(api_key="test", base_url=fake_server.base_url, max_retries=0)


def test_is_valid_model_accepts_model_added_after_first_lookup(fake_server, client):
    assert client.is_valid_model("gpt-4o-8k")
    assert not client.is_valid_model("gpt-5.6-8k")

    fake_server.model_costs["gpt-5.6-8k"] = 2

    assert client.is_valid_model("gpt-5.6-8k")
    assert "gpt-5.6-8k" in client.agent_models


def test_model_cost_reflects_cost_changed_after_first_lookup(fake_server, client):
    assert client.model_cost("gpt-4o-8k") == 1

    fake_server.model_costs["gpt-4o-8k"] = 3

    assert client.model_cost("gpt-4o-8k") == 3

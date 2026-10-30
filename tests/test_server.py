# SPDX-License-Identifier: Apache-2.0
"""Server integration tests on the toy model (FastAPI TestClient)."""

import json

import pytest

from agrimlxo.server import ServerState, create_app
from agrimlxo.loader import LoadedModel
from agrimlxo.engine import Engine
from agrimlxo.moe_expansion import MoERuntime, MoEExpansionConfig

from test_engine_toy import ToyTokenizer, build_toy_model, PROMPT


class Args:
    model = "toy-qwen3moe"
    host = "127.0.0.1"
    port = 8080
    api_key = None
    ctx_size = 256
    parallel = 2
    n_predict = 16
    temp = 0.8
    top_k = 40
    top_p = 0.95
    min_p = 0.05
    xtc_probability = 0.0
    xtc_threshold = 0.1
    repeat_penalty = 1.0
    repeat_last_n = 64
    presence_penalty = 0.0
    frequency_penalty = 0.0
    seed = 42  # deterministic sampling (random draws can hit the toy eos)
    reasoning_budget = 0
    reasoning_format = "auto"


@pytest.fixture()
def client(tmp_path):
    d = build_toy_model(tmp_path)
    rt = MoERuntime(4, True, MoEExpansionConfig(max_experts=8, threshold=0.8))
    rt.setup(True, 2, 8)
    loaded = LoadedModel(d, expansion=rt)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=256, parallel=2)
    state = ServerState(Args(), engine, ToyTokenizer())
    from fastapi.testclient import TestClient

    with TestClient(create_app(state)) as c:
        yield c


def test_health_and_version(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "version" in client.get("/version").json()


def test_props_reports_moe(client):
    props = client.get("/props").json()
    assert props["moe_expansion"]["expansion_active"] is True
    assert props["moe_expansion"]["experts_per_token_max"] == 8
    assert props["moe_expansion"]["native_top_k"] == 2
    assert props["total_slots"] == 2


def test_models_and_metrics(client):
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "toy-qwen3moe"
    metrics = client.get("/metrics").text
    assert "agrimlxo_requests_total" in metrics
    assert "agrimlxo_moe_expansion" in metrics


def test_chat_completion_non_stream(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "ciao"}], "max_tokens": 4},
    )
    assert r.status_code == 200
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert body["usage"]["completion_tokens"] >= 1


def test_chat_completion_stream(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "ciao"}],
            "max_tokens": 4,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(line[6:])
    assert events[-1] == "[DONE]" or json.loads(events[-1]).get("usage")
    chunks = [json.loads(e) for e in events if e != "[DONE]"]
    final = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert final, "expected a finish_reason chunk"
    usage = [c for c in chunks if "usage" in c]
    assert usage and usage[-1]["usage"]["completion_tokens"] >= 1


def test_completions_endpoint(client):
    r = client.post(
        "/v1/completions",
        json={"prompt": "1 2 3 4 5", "max_tokens": 4, "seed": 42},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["text"] is not None
    assert body["usage"]["completion_tokens"] >= 1


def test_slots_endpoint(client):
    slots = client.get("/slots").json()["slots"]
    assert len(slots) == 2 and slots[0]["busy"] is False

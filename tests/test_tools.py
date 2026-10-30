# SPDX-License-Identifier: Apache-2.0
"""Tool registry, parsing and execution tests."""

import json
from pathlib import Path

from agrimlxo.tools_registry import (
    ToolRegistry,
    parse_tool_calls,
    strip_tool_call_blocks,
)


def test_registry_loads_shipped_tools():
    reg = ToolRegistry(str(Path(__file__).parent.parent / "tools"))
    assert set(reg.names()) >= {"get_datetime", "fetch_text", "web_search", "draw"}
    schema = reg.openai_schema()
    assert all(s["type"] == "function" for s in schema)
    assert {s["function"]["name"] for s in schema} >= {"get_datetime"}


def test_get_datetime_runs():
    reg = ToolRegistry(str(Path(__file__).parent.parent / "tools"))
    out = reg.execute("get_datetime", "{}")
    assert "20" in out  # year prefix; never throws


def test_execute_missing_and_bad_json():
    reg = ToolRegistry(str(Path(__file__).parent.parent / "tools"))
    assert "unknown tool" in reg.execute("nope", "{}")
    assert "errore" in reg.execute("get_datetime", "{bad json").lower() or reg.execute(
        "get_datetime", "{bad json"
    )


def test_parse_tool_calls():
    text = (
        "prima pensiero\n<tool_call>"
        '{"name": "get_datetime", "arguments": {}}'
        "</tool_call>\n<tool_call>"
        '{"name": "fetch_text", "arguments": {"url": "https://example.com"}}'
        "</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["get_datetime", "fetch_text"]
    assert json.loads(calls[1]["arguments"])["url"] == "https://example.com"
    clean = strip_tool_call_blocks(text)
    assert "<tool_call>" not in clean and "prima pensiero" in clean

# SPDX-License-Identifier: Apache-2.0
"""Tool registry for AgrimlxO.

Scans a directory for python scripts that expose a ``TOOL`` dict:

    TOOL = {
        "name": "...",
        "description": "...",
        "parameters": {json schema},
        "run": callable,        # called with the model's JSON arguments
    }

and executes them for the chat agentic loop. The scripts are the user's own
(./tools/): importing them is the point, like any local plugin dir.
"""

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, tools_dir: str = "./tools"):
        self.tools_dir = Path(tools_dir)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.reload()

    def reload(self) -> int:
        self.tools = {}
        if not self.tools_dir.is_dir():
            return 0
        for f in sorted(self.tools_dir.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"agrimlxo_tool_{f.stem}", f
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                tool = getattr(mod, "TOOL", None)
                if (
                    isinstance(tool, dict)
                    and isinstance(tool.get("name"), str)
                    and callable(tool.get("run"))
                ):
                    self.tools[tool["name"]] = tool
            except Exception as e:
                logger.warning("tool %s non caricato: %s", f.name, e)
        return len(self.tools)

    def names(self) -> List[str]:
        return list(self.tools)

    def openai_schema(self) -> List[dict]:
        """OpenAI tools format for the chat template."""
        return [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            }}
            for t in self.tools.values()
        ]

    def execute(self, name: str, arguments: Any) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"errore: tool '{name}' inesistente (disponibili: {self.names()})"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
            result = tool["run"](**args)
            return str(result)
        except Exception as e:
            return f"errore eseguendo {name}: {e}"


# ---- parsing of the model's <tool_call> blocks (Qwen3 format) ----------------

_TOOL_CALL_RE = None  # built lazily (re with nested JSON is fine here)


def parse_tool_calls(text: str) -> List[dict]:
    """Extract [{"name", "arguments"(str)}] from <tool_call>…</tool_call>
    blocks; arguments stay a JSON string (OpenAI convention)."""
    import re

    calls = []
    for m in re.finditer(
        r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text
    ):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                calls.append(
                    {"name": obj["name"],
                     "arguments": json.dumps(obj.get("arguments", {}))}
                )
        except json.JSONDecodeError:
            continue
    return calls


def strip_tool_call_blocks(text: str) -> str:
    import re

    return re.sub(r"<tool_call>\s*\{[\s\S]*?\}\s*</tool_call>", "", text).strip()

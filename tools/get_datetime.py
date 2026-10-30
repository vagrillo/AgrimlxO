# SPDX-License-Identifier: Apache-2.0
"""Tool: data e ora esatte del server."""

from datetime import datetime

TOOL = {
    "name": "get_datetime",
    "description": "Returns the server's current local date and time.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def run(**kwargs) -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A, %Z)")

TOOL["run"] = run

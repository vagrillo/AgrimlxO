# SPDX-License-Identifier: Apache-2.0
"""Tool: data e ora esatte del server."""

from datetime import datetime

TOOL = {
    "name": "get_datetime",
    "description": "Ritorna la data e l'ora esatte correnti del server (locale).",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def run(**kwargs) -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A, %Z)")

TOOL["run"] = run

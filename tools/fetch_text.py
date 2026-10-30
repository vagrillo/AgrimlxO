# SPDX-License-Identifier: Apache-2.0
"""Tool: scarica una pagina web e ritorna il solo contenuto testuale."""

import re
import urllib.request

TOOL = {
    "name": "fetch_text",
    "description": "Scarica una pagina web (URL) e ritorna il solo testo, senza tag HTML.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL http(s) della pagina"},
            "max_chars": {"type": "integer", "description": "tronca il testo a N caratteri (default 8000)"},
        },
        "required": ["url"],
    },
}

_TAG_RE = re.compile(r"<(script|style)[\s\S]*?</\1>", re.I)
_TAG_OPEN = re.compile(r"<[^>]+>")
_WS = re.compile(r"\n{3,}")


def run(url: str = "", max_chars: int = 8000, **kwargs) -> str:
    if not url:
        return "errore: url mancante"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (AgrimlxO tool)"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        html = r.read().decode(r.headers.get_content_charset() or "utf-8", "replace")
    html = _TAG_RE.sub(" ", html)
    text = _TAG_OPEN.sub(" ", html)
    text = re.sub(r"[ \t]+", " ", text)
    text = _WS.sub("\n\n", text).strip()
    return text[: int(max_chars or 8000)]

TOOL["run"] = run

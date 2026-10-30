# SPDX-License-Identifier: Apache-2.0
"""Tool: ricerca web via DuckDuckGo (HTML, senza API key)."""

import re
import urllib.parse
import urllib.request

TOOL = {
    "name": "web_search",
    "description": "Cerca sul web e ritorna i primi risultati (titolo, URL, snippet).",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "query di ricerca"},
            "n": {"type": "integer", "description": "numero di risultati (default 5)"},
        },
        "required": ["query"],
    },
}

_RESULT = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>[\s\S]*?'
    r'class="result__snippet"[^>]*>([\s\S]*?)</a>',
    re.I,
)
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return _TAG.sub("", s).replace("&amp;", "&").replace("&#x27;", "'").strip()


def run(query: str = "", n: int = 5, **kwargs) -> str:
    if not query:
        return "errore: query mancante"
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (AgrimlxO tool)"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        html = r.read().decode("utf-8", "replace")
    out = []
    for m in _RESULT.finditer(html):
        link = urllib.parse.unquote(m.group(1))
        if link.startswith("//duckduckgo.com/l/?uddg="):
            link = link.split("uddg=")[1].split("&")[0]
            link = urllib.parse.unquote(link)
        out.append(f"- {_clean(m.group(2))}\n  {link}\n  {_clean(m.group(3))[:200]}")
        if len(out) >= int(n or 5):
            break
    return "\n".join(out) if out else "nessun risultato"

TOOL["run"] = run

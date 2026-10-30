# SPDX-License-Identifier: Apache-2.0
"""Tool: draw — skill di disegno SVG in stile cartoon giapponese.

The model calls it when the user asks to draw/generate an image; the
call result is the style skill the model follows in the next phase to
produce the SVG (rendered inline by the webui).
"""

SKILL = (
    "You are an assistant able to draw: if the user asks you to draw "
    "something or generate an image, realize it as SVG code, using a "
    "Japanese cartoon model style, lots of details, bright colors, thick lines."
)

TOOL = {
    "name": "draw",
    "description": (
        "Call this when the user asks to draw something or generate an "
        "image: you will receive the style skill to follow, then produce "
        "the drawing as complete SVG code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "what to draw, with the details the user asked for",
            }
        },
        "required": ["subject"],
    },
}


def run(subject: str = "", **kwargs) -> str:
    return (
        f"{SKILL}\n\nOra disegna: {subject or 'il soggetto richiesto'}.\n"
        "Answer ONLY with the complete SVG code (starting with <svg>)."
    )


TOOL["run"] = run

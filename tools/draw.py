# SPDX-License-Identifier: Apache-2.0
"""Tool: draw — skill di disegno SVG in stile cartoon giapponese.

Il modello la chiama quando l'utente chiede di disegnare/generare
un'immagine; il risultato della chiamata è lo skill di stile che il modello
segue nella fase successiva per produrre l'SVG (reso inline dalla webui).
"""

SKILL = (
    "Sei un assistente in grado di disegnare, se l'utente ti chiede di "
    "disegnare qualcosa o di generare un immagine, realizzala in codice SVG, "
    "usa uno stile modello cartoon giapponese, molti dettagli, colori accesi, "
    "linee grosse."
)

TOOL = {
    "name": "draw",
    "description": (
        "Chiamala quando l'utente chiede di disegnare qualcosa o di generare "
        "un'immagine: riceverai lo skill di stile da seguire e poi "
        "producrai il disegno come codice SVG completo."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "cosa disegnare, con i dettagli richiesti dall'utente",
            }
        },
        "required": ["subject"],
    },
}


def run(subject: str = "", **kwargs) -> str:
    return (
        f"{SKILL}\n\nOra disegna: {subject or 'il soggetto richiesto'}.\n"
        "Rispondi SOLO con il codice SVG completo (inizia con <svg>)."
    )


TOOL["run"] = run

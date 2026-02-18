# src/asciart/renderers/text.py
from __future__ import annotations
from asciart.models import AsciiArt


def render_text(art: AsciiArt) -> str:
    """Render AsciiArt as plain text with no color codes."""
    lines = []
    for row in art.cells:
        lines.append("".join(cell.char for cell in row))
    return "\n".join(lines)

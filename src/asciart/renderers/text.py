from __future__ import annotations
from asciart.models import AsciiArt


def _render_text_arrays(art: AsciiArt) -> str:
    """Fast path: read directly from char_array and char_map."""
    char_array = art.char_array
    char_map = art.char_map or ""
    map_len = len(char_map)
    lines = []
    for y in range(art.height):
        row_indices = char_array[y]  # type: ignore[index]
        chars = []
        for x in range(art.width):
            idx = int(row_indices[x])
            chars.append(char_map[idx] if idx < map_len else " ")
        lines.append("".join(chars))
    return "\n".join(lines)


def render_text(art: AsciiArt) -> str:
    """Render AsciiArt as plain text with no color codes."""
    if art.char_array is not None:
        return _render_text_arrays(art)
    # Legacy fallback
    lines = []
    for row in art.cells:
        lines.append("".join(cell.char for cell in row))
    return "\n".join(lines)

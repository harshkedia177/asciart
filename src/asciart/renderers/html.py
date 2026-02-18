# src/asciart/renderers/html.py
from __future__ import annotations
from html import escape
from asciart.models import AsciiArt

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background: #1e1e1e; display: flex; justify-content: center; padding: 20px; }}
  pre {{ font-family: 'Courier New', monospace; font-size: 10px; line-height: 1.0; color: #ccc; }}
</style>
</head>
<body>
<pre>{body}</pre>
</body>
</html>"""


def _render_html_arrays(art: AsciiArt, title: str = "ASCII Art") -> str:
    """Fast path: read directly from arrays."""
    char_array = art.char_array
    char_map = art.char_map or ""
    map_len = len(char_map)
    fg_array = art.fg_array
    bg_array = art.bg_array
    lines = []
    for y in range(art.height):
        parts = []
        for x in range(art.width):
            idx = int(char_array[y, x])  # type: ignore[index]
            raw_char = char_map[idx] if idx < map_len else " "
            char = escape(raw_char)
            if char == " ":
                char = "&nbsp;"
            if fg_array is not None:
                r, g, b = int(fg_array[y, x, 0]), int(fg_array[y, x, 1]), int(fg_array[y, x, 2])  # type: ignore[index]
                style = f"color:rgb({r},{g},{b})"
                if bg_array is not None:
                    br, bg_val, bb = int(bg_array[y, x, 0]), int(bg_array[y, x, 1]), int(bg_array[y, x, 2])  # type: ignore[index]
                    style += f";background:rgb({br},{bg_val},{bb})"
                parts.append(f'<span style="{style}">{char}</span>')
            else:
                parts.append(char)
        lines.append("".join(parts))
    body = "\n".join(lines)
    return _HTML_TEMPLATE.format(title=escape(title), body=body)


def render_html(art: AsciiArt, title: str = "ASCII Art") -> str:
    if art.char_array is not None:
        return _render_html_arrays(art, title)
    # Legacy fallback
    lines = []
    for row in art.cells:
        parts = []
        for cell in row:
            char = escape(cell.char)
            if char == " ":
                char = "&nbsp;"
            if cell.fg is not None:
                r, g, b = cell.fg
                style = f"color:rgb({r},{g},{b})"
                if cell.bg is not None:
                    br, bg, bb = cell.bg
                    style += f";background:rgb({br},{bg},{bb})"
                parts.append(f'<span style="{style}">{char}</span>')
            else:
                parts.append(char)
        lines.append("".join(parts))
    body = "\n".join(lines)
    return _HTML_TEMPLATE.format(title=escape(title), body=body)

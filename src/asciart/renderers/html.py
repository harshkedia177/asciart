# src/asciart/renderers/html.py
from __future__ import annotations
from html import escape
from asciart.models import AsciiArt


def render_html(art: AsciiArt, title: str = "ASCII Art") -> str:
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
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
  body {{ background: #1e1e1e; display: flex; justify-content: center; padding: 20px; }}
  pre {{ font-family: 'Courier New', monospace; font-size: 10px; line-height: 1.0; color: #ccc; }}
</style>
</head>
<body>
<pre>{body}</pre>
</body>
</html>"""

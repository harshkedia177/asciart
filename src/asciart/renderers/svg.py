from __future__ import annotations
from html import escape
from asciart.models import AsciiArt

CHAR_WIDTH = 7.2
CHAR_HEIGHT = 14

_SVG_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" style="background:#1e1e1e">
  <style>text {{ font-family: 'Courier New', monospace; font-size: 12px; }}</style>
  {elements}
</svg>"""


def _render_svg_arrays(art: AsciiArt) -> str:
    """Fast path: read directly from arrays."""
    svg_w = art.width * CHAR_WIDTH + 20
    svg_h = art.height * CHAR_HEIGHT + 20
    char_array = art.char_array
    char_map = art.char_map or ""
    map_len = len(char_map)
    fg_array = art.fg_array
    text_elements = []
    for y in range(art.height):
        x_pos = 10
        y_pos = 10 + (y + 1) * CHAR_HEIGHT
        line_chars = ""
        current_color = None
        for x in range(art.width):
            idx = int(char_array[y, x])  # type: ignore[index]
            char = char_map[idx] if idx < map_len else " "
            if fg_array is not None:
                r, g, b = int(fg_array[y, x, 0]), int(fg_array[y, x, 1]), int(fg_array[y, x, 2])  # type: ignore[index]
                color = f"rgb({r},{g},{b})"
            else:
                color = "#cccccc"
            if color != current_color:
                if line_chars:
                    text_elements.append(
                        f'<text x="{x_pos}" y="{y_pos}" fill="{current_color}">'
                        f"{escape(line_chars)}</text>"
                    )
                    x_pos += len(line_chars) * CHAR_WIDTH
                line_chars = char
                current_color = color
            else:
                line_chars += char
        if line_chars:
            text_elements.append(
                f'<text x="{x_pos}" y="{y_pos}" fill="{current_color}">'
                f"{escape(line_chars)}</text>"
            )
    elements = "\n  ".join(text_elements)
    return _SVG_TEMPLATE.format(svg_w=svg_w, svg_h=svg_h, elements=elements)


def render_svg(art: AsciiArt) -> str:
    if art.char_array is not None:
        return _render_svg_arrays(art)
    # Legacy fallback
    svg_w = art.width * CHAR_WIDTH + 20
    svg_h = art.height * CHAR_HEIGHT + 20
    text_elements = []
    for y, row in enumerate(art.cells):
        x_pos = 10
        y_pos = 10 + (y + 1) * CHAR_HEIGHT
        line_chars = ""
        current_color = None
        for cell in row:
            color = f"rgb({cell.fg[0]},{cell.fg[1]},{cell.fg[2]})" if cell.fg else "#cccccc"
            if color != current_color:
                if line_chars:
                    text_elements.append(
                        f'<text x="{x_pos}" y="{y_pos}" fill="{current_color}">'
                        f"{escape(line_chars)}</text>"
                    )
                    x_pos += len(line_chars) * CHAR_WIDTH
                line_chars = cell.char
                current_color = color
            else:
                line_chars += cell.char
        if line_chars:
            text_elements.append(
                f'<text x="{x_pos}" y="{y_pos}" fill="{current_color}">'
                f"{escape(line_chars)}</text>"
            )
    elements = "\n  ".join(text_elements)
    return _SVG_TEMPLATE.format(svg_w=svg_w, svg_h=svg_h, elements=elements)

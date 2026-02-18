# src/asciart/renderers/terminal.py
from __future__ import annotations
from asciart.models import AsciiArt

RESET = "\033[0m"


def _truecolor_fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _truecolor_bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def _ansi256_fg(r: int, g: int, b: int) -> str:
    """Map RGB to nearest xterm-256 color."""
    if abs(r - g) < 10 and abs(g - b) < 10:
        if r < 8:
            idx = 16
        elif r > 248:
            idx = 231
        else:
            idx = round((r - 8) / 247 * 24) + 232
    else:
        ri = round(r / 255 * 5)
        gi = round(g / 255 * 5)
        bi = round(b / 255 * 5)
        idx = 16 + 36 * ri + 6 * gi + bi
    return f"\033[38;5;{idx}m"


def render_ansi(art: AsciiArt, use_256: bool = False) -> str:
    """Render AsciiArt with ANSI color escape codes."""
    lines = []
    for row in art.cells:
        parts = []
        for cell in row:
            codes = ""
            if cell.fg is not None:
                if use_256:
                    codes += _ansi256_fg(*cell.fg)
                else:
                    codes += _truecolor_fg(*cell.fg)
            if cell.bg is not None:
                codes += _truecolor_bg(*cell.bg)
            parts.append(f"{codes}{cell.char}")
        lines.append("".join(parts) + RESET)
    return "\n".join(lines)

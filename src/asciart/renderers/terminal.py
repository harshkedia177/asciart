from __future__ import annotations
from asciart.core.color_space import nearest_ansi256_perceptual
from asciart.models import AsciiArt

RESET = "\033[0m"


def _truecolor_fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _truecolor_bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def _ansi256_fg_perceptual(r: int, g: int, b: int) -> str:
    """Map RGB to nearest xterm-256 color using CIELAB perceptual distance."""
    idx = nearest_ansi256_perceptual(r, g, b)
    return f"\033[38;5;{idx}m"


def _render_ansi_arrays(art: AsciiArt, use_256: bool = False) -> str:
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
            char = char_map[idx] if idx < map_len else " "
            codes = ""
            if fg_array is not None:
                r, g, b = int(fg_array[y, x, 0]), int(fg_array[y, x, 1]), int(fg_array[y, x, 2])  # type: ignore[index]
                if use_256:
                    codes += _ansi256_fg_perceptual(r, g, b)
                else:
                    codes += _truecolor_fg(r, g, b)
            if bg_array is not None:
                r, g, b = int(bg_array[y, x, 0]), int(bg_array[y, x, 1]), int(bg_array[y, x, 2])  # type: ignore[index]
                codes += _truecolor_bg(r, g, b)
            parts.append(f"{codes}{char}")
        lines.append("".join(parts) + RESET)
    return "\n".join(lines)


def render_ansi(art: AsciiArt, use_256: bool = False) -> str:
    """Render AsciiArt with ANSI color escape codes."""
    if art.char_array is not None:
        return _render_ansi_arrays(art, use_256)
    # Legacy fallback
    lines = []
    for row in art.cells:
        parts = []
        for cell in row:
            codes = ""
            if cell.fg is not None:
                if use_256:
                    codes += _ansi256_fg_perceptual(*cell.fg)
                else:
                    codes += _truecolor_fg(*cell.fg)
            if cell.bg is not None:
                codes += _truecolor_bg(*cell.bg)
            parts.append(f"{codes}{cell.char}")
        lines.append("".join(parts) + RESET)
    return "\n".join(lines)

from __future__ import annotations
import io
from PIL import Image, ImageDraw, ImageFont
from asciart.models import AsciiArt

CHAR_WIDTH = 8
CHAR_HEIGHT = 16
BG_COLOR = (30, 30, 30)
DEFAULT_FG = (204, 204, 204)


def _render_png_arrays(art: AsciiArt) -> bytes:
    """Fast path: read directly from arrays."""
    img_w = art.width * CHAR_WIDTH
    img_h = art.height * CHAR_HEIGHT
    img = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Courier", CHAR_HEIGHT)
    except (OSError, IOError):
        font = ImageFont.load_default()
    char_array = art.char_array
    char_map = art.char_map or ""
    map_len = len(char_map)
    fg_array = art.fg_array
    bg_array = art.bg_array
    for y in range(art.height):
        for x in range(art.width):
            idx = int(char_array[y, x])  # type: ignore[index]
            char = char_map[idx] if idx < map_len else " "
            px = x * CHAR_WIDTH
            py = y * CHAR_HEIGHT
            if bg_array is not None:
                br, bg_val, bb = int(bg_array[y, x, 0]), int(bg_array[y, x, 1]), int(bg_array[y, x, 2])  # type: ignore[index]
                draw.rectangle([px, py, px + CHAR_WIDTH, py + CHAR_HEIGHT], fill=(br, bg_val, bb))
            if fg_array is not None:
                fg = (int(fg_array[y, x, 0]), int(fg_array[y, x, 1]), int(fg_array[y, x, 2]))  # type: ignore[index]
            else:
                fg = DEFAULT_FG
            draw.text((px, py), char, fill=fg, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_png(art: AsciiArt) -> bytes:
    if art.char_array is not None:
        return _render_png_arrays(art)
    # Legacy fallback
    img_w = art.width * CHAR_WIDTH
    img_h = art.height * CHAR_HEIGHT
    img = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Courier", CHAR_HEIGHT)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for y, row in enumerate(art.cells):
        for x, cell in enumerate(row):
            fg = cell.fg if cell.fg else DEFAULT_FG
            px = x * CHAR_WIDTH
            py = y * CHAR_HEIGHT
            if cell.bg:
                draw.rectangle([px, py, px + CHAR_WIDTH, py + CHAR_HEIGHT], fill=cell.bg)
            draw.text((px, py), cell.char, fill=fg, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# src/asciart/renderers/image.py
from __future__ import annotations
import io
from PIL import Image, ImageDraw, ImageFont
from asciart.models import AsciiArt

CHAR_WIDTH = 8
CHAR_HEIGHT = 16
BG_COLOR = (30, 30, 30)
DEFAULT_FG = (204, 204, 204)


def render_png(art: AsciiArt) -> bytes:
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

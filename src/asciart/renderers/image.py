from __future__ import annotations
import io
from PIL import Image, ImageDraw, ImageFont
from asciart.models import AsciiArt

CHAR_WIDTH = 8
CHAR_HEIGHT = 16
BG_COLOR = (30, 30, 30)
DEFAULT_FG = (204, 204, 204)


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Courier", CHAR_HEIGHT)
    except (OSError, IOError):
        return ImageFont.load_default()


def _create_canvas(art: AsciiArt) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (art.width * CHAR_WIDTH, art.height * CHAR_HEIGHT), BG_COLOR)
    return img, ImageDraw.Draw(img)


def _image_to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _draw_arrays(draw: ImageDraw.ImageDraw, art: AsciiArt, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> None:
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


def _draw_cells(draw: ImageDraw.ImageDraw, art: AsciiArt, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> None:
    for y, row in enumerate(art.cells):
        for x, cell in enumerate(row):
            fg = cell.fg if cell.fg else DEFAULT_FG
            px = x * CHAR_WIDTH
            py = y * CHAR_HEIGHT
            if cell.bg:
                draw.rectangle([px, py, px + CHAR_WIDTH, py + CHAR_HEIGHT], fill=cell.bg)
            draw.text((px, py), cell.char, fill=fg, font=font)


def render_png(art: AsciiArt) -> bytes:
    img, draw = _create_canvas(art)
    font = _load_font()
    if art.char_array is not None:
        _draw_arrays(draw, art, font)
    else:
        _draw_cells(draw, art, font)
    return _image_to_png(img)

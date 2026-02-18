# tests/test_braille.py
from PIL import Image
from asciart.core.engine import convert
from asciart.models import ConvertOptions, Mode, ColorMode


def test_braille_mode_uses_braille_chars():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    opts = ConvertOptions(width=20, mode=Mode.BRAILLE, color=ColorMode.NONE)
    result = convert(img, opts)
    for row in result.cells:
        for cell in row:
            assert 0x2800 <= ord(cell.char) <= 0x28FF


def test_braille_white_is_empty():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    opts = ConvertOptions(width=20, mode=Mode.BRAILLE, color=ColorMode.NONE)
    result = convert(img, opts)
    for row in result.cells:
        for cell in row:
            assert cell.char == "\u2800"


def test_braille_black_is_full():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    opts = ConvertOptions(width=20, mode=Mode.BRAILLE, color=ColorMode.NONE)
    result = convert(img, opts)
    for row in result.cells:
        for cell in row:
            assert cell.char == "\u28FF"


def test_blocks_mode_uses_block_chars():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    opts = ConvertOptions(width=20, mode=Mode.BLOCKS, color=ColorMode.NONE)
    result = convert(img, opts)
    block_chars = set(" \u2591\u2592\u2593\u2588")
    for row in result.cells:
        for cell in row:
            assert cell.char in block_chars


def test_halfblock_mode_has_fg_and_bg():
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    opts = ConvertOptions(width=20, mode=Mode.HALFBLOCK, color=ColorMode.TRUECOLOR)
    result = convert(img, opts)
    for row in result.cells:
        for cell in row:
            assert cell.char == "\u2580"
            assert cell.fg is not None
            assert cell.bg is not None

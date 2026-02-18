# tests/test_engine.py
from asciart.core.engine import convert
from asciart.models import ConvertOptions, ColorMode, AsciiArt


def test_convert_returns_ascii_art(white_image):
    opts = ConvertOptions(width=10)
    result = convert(white_image, opts)
    assert isinstance(result, AsciiArt)
    assert result.width == 10
    assert len(result.cells) > 0
    assert len(result.cells[0]) == 10


def test_white_image_maps_to_lightest_char(white_image):
    opts = ConvertOptions(width=10, chars=" .:#", color=ColorMode.NONE)
    result = convert(white_image, opts)
    for row in result.cells:
        for cell in row:
            assert cell.char == " "


def test_black_image_maps_to_darkest_char(black_image):
    opts = ConvertOptions(width=10, chars=" .:#", color=ColorMode.NONE)
    result = convert(black_image, opts)
    for row in result.cells:
        for cell in row:
            assert cell.char == "#"


def test_invert_swaps_mapping(white_image):
    opts = ConvertOptions(width=10, chars=" .:#", color=ColorMode.NONE, invert=True)
    result = convert(white_image, opts)
    for row in result.cells:
        for cell in row:
            assert cell.char == "#"


def test_gradient_uses_full_ramp(gradient_image):
    opts = ConvertOptions(width=256, chars=" .:-=+*#%@", color=ColorMode.NONE)
    result = convert(gradient_image, opts)
    chars_used = {cell.char for row in result.cells for cell in row}
    assert len(chars_used) >= 8


def test_color_mode_attaches_rgb(small_color_image):
    opts = ConvertOptions(width=4, color=ColorMode.TRUECOLOR)
    result = convert(small_color_image, opts)
    for row in result.cells:
        for cell in row:
            assert cell.fg is not None
            assert len(cell.fg) == 3


def test_no_color_mode_no_rgb(small_color_image):
    opts = ConvertOptions(width=4, color=ColorMode.NONE)
    result = convert(small_color_image, opts)
    for row in result.cells:
        for cell in row:
            assert cell.fg is None


def test_aspect_ratio_correction(white_image):
    opts = ConvertOptions(width=20, font_ratio=0.5)
    result = convert(white_image, opts)
    assert result.height == 10

from asciart.models import AsciiArt, Cell
from asciart.renderers.terminal import render_ansi
from asciart.renderers.text import render_text


def _make_art():
    cells = [
        [Cell(char="@", fg=(255, 0, 0)), Cell(char="#", fg=(0, 255, 0))],
        [Cell(char=".", fg=(0, 0, 255)), Cell(char=" ")],
    ]
    return AsciiArt(width=2, height=2, cells=cells)


def test_render_text_plain():
    art = _make_art()
    result = render_text(art)
    assert result == "@#\n. "


def test_render_text_no_ansi():
    art = _make_art()
    result = render_text(art)
    assert "\033" not in result


def test_render_ansi_contains_escape_codes():
    art = _make_art()
    result = render_ansi(art)
    assert "\033[" in result


def test_render_ansi_contains_reset():
    art = _make_art()
    result = render_ansi(art)
    assert "\033[0m" in result


def test_render_ansi_contains_characters():
    art = _make_art()
    result = render_ansi(art)
    assert "@" in result
    assert "#" in result


def test_render_text_no_color_cells():
    cells = [[Cell(char="X"), Cell(char="Y")]]
    art = AsciiArt(width=2, height=1, cells=cells)
    result = render_text(art)
    assert result == "XY"

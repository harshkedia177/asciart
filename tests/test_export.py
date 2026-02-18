from asciart.models import AsciiArt, Cell
from asciart.renderers.html import render_html
from asciart.renderers.svg import render_svg
from asciart.renderers.image import render_png

def _make_colored_art():
    cells = [
        [Cell(char="@", fg=(255, 0, 0)), Cell(char="#", fg=(0, 255, 0)), Cell(char="*", fg=(0, 0, 255))],
        [Cell(char=".", fg=(128, 128, 0)), Cell(char=":", fg=(0, 128, 128)), Cell(char=" ")],
    ]
    return AsciiArt(width=3, height=2, cells=cells)

def test_html_is_valid_html():
    art = _make_colored_art()
    html = render_html(art)
    assert "<html" in html.lower()
    assert "<pre" in html.lower()
    assert "</html>" in html.lower()
    assert "@" in html
    assert "#" in html

def test_html_contains_color_styles():
    art = _make_colored_art()
    html = render_html(art)
    assert "color:" in html or "rgb(" in html

def test_svg_is_valid_svg():
    art = _make_colored_art()
    svg = render_svg(art)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "<text" in svg

def test_png_returns_bytes():
    art = _make_colored_art()
    data = render_png(art)
    assert isinstance(data, bytes)
    assert data[:4] == b"\x89PNG"

def test_png_nonzero_size():
    art = _make_colored_art()
    data = render_png(art)
    assert len(data) > 100

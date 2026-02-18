from asciart.models import AsciiArt, Cell, ConvertOptions, Mode, ColorMode, DitherMode


def test_cell_defaults():
    cell = Cell(char="@")
    assert cell.char == "@"
    assert cell.fg is None
    assert cell.bg is None


def test_cell_with_color():
    cell = Cell(char="#", fg=(255, 0, 0))
    assert cell.fg == (255, 0, 0)


def test_ascii_art_creation():
    cells = [[Cell(char="@"), Cell(char="#")], [Cell(char=".", fg=(128, 128, 128)), Cell(char=" ")]]
    art = AsciiArt(width=2, height=2, cells=cells)
    assert art.width == 2
    assert art.height == 2
    assert art.cells[0][0].char == "@"
    assert art.cells[1][0].fg == (128, 128, 128)


def test_convert_options_defaults():
    opts = ConvertOptions()
    assert opts.width == 80
    assert opts.mode == Mode.ASCII
    assert opts.color == ColorMode.AUTO
    assert opts.dither == DitherMode.NONE
    assert opts.contrast == 1.0
    assert opts.font_ratio == 0.5

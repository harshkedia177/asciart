# asciart Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python CLI tool that converts images to stunning ASCII art with multiple rendering modes, an interactive TUI, and multiple export formats.

**Architecture:** Core engine (pure computation, zero I/O) produces an `AsciiArt` grid from image pixels + options. Renderers convert that grid to terminal/HTML/PNG/SVG/text output. Two interfaces: a one-shot Typer CLI and an interactive Textual TUI with live preview and sliders.

**Tech Stack:** Python 3.11+, uv, Pillow, NumPy, Typer, Textual, Rich, pytest, ruff

**Design Doc:** `docs/plans/2026-02-18-asciart-cli-design.md`

---

## Task 1: Project Scaffolding with uv

**Files:**
- Create: `pyproject.toml`
- Create: `src/asciart/__init__.py`
- Create: `src/asciart/models.py`
- Create: `src/asciart/core/__init__.py`
- Create: `src/asciart/core/ramps.py`
- Create: `src/asciart/renderers/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`

**Step 1: Initialize project with uv**

```bash
cd /Users/harshkedia/Custom/Personal/asci-art
uv init --name asciart --package --python ">=3.11"
```

Then update `pyproject.toml` to use `src` layout:

```toml
[project]
name = "asciart"
version = "0.1.0"
description = "Convert images to stunning ASCII art"
requires-python = ">=3.11"
dependencies = [
    "pillow>=10.0",
    "numpy>=1.24",
    "typer>=0.9",
    "textual>=0.80",
    "rich>=13.0",
]

[project.scripts]
asciart = "asciart.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.5",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Install dependencies**

```bash
uv sync
```

**Step 3: Create directory structure**

```bash
mkdir -p src/asciart/core src/asciart/renderers tests
```

**Step 4: Write models.py — the core data structures**

```python
# src/asciart/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    ASCII = "ascii"
    BLOCKS = "blocks"
    BRAILLE = "braille"
    HALFBLOCK = "halfblock"


class ColorMode(str, Enum):
    NONE = "none"
    ANSI256 = "256"
    TRUECOLOR = "truecolor"
    AUTO = "auto"


class DitherMode(str, Enum):
    NONE = "none"
    FLOYD_STEINBERG = "floyd-steinberg"
    ORDERED = "ordered"


class OutputFormat(str, Enum):
    TEXT = "text"
    ANSI = "ansi"
    HTML = "html"
    PNG = "png"
    SVG = "svg"


@dataclass
class Cell:
    """A single character cell in the ASCII art grid."""

    char: str
    fg: tuple[int, int, int] | None = None
    bg: tuple[int, int, int] | None = None


@dataclass
class AsciiArt:
    """The intermediate representation — output of core engine, input to renderers."""

    width: int
    height: int
    cells: list[list[Cell]] = field(default_factory=list)


@dataclass
class ConvertOptions:
    """All settings for a conversion. Every slider/flag maps to a field here."""

    width: int = 80
    height: int | None = None
    mode: Mode = Mode.ASCII
    chars: str = " .:-=+*#%@"
    color: ColorMode = ColorMode.AUTO
    dither: DitherMode = DitherMode.NONE
    invert: bool = False
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    font_ratio: float = 0.5
    edge_detection: bool = False
    edge_threshold: float = 50.0
```

**Step 5: Write ramps.py — character ramp definitions**

```python
# src/asciart/core/ramps.py

# Short 10-char ramp — good general purpose
STANDARD = " .:-=+*#%@"

# Minimal 4-char ramp — retro feel
MINIMAL = " .:#"

# Paul Bourke's 70-char ramp — finest granularity
DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Unicode block characters
BLOCKS = " ░▒▓█"
```

**Step 6: Write and run the first test**

```python
# tests/test_models.py
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
```

```bash
uv run pytest tests/test_models.py -v
```

Expected: all 4 tests PASS.

**Step 7: Create `__init__.py` files**

```python
# src/asciart/__init__.py
"""asciart — Convert images to stunning ASCII art."""
```

```python
# src/asciart/core/__init__.py
```

```python
# src/asciart/renderers/__init__.py
```

```python
# tests/__init__.py
```

**Step 8: Git init and first commit**

```bash
git init
git add .
git commit -m "feat: project scaffolding with uv, models, and ramps"
```

---

## Task 2: Core Engine — Basic Brightness Mapping

**Files:**
- Create: `src/asciart/core/preprocess.py`
- Create: `src/asciart/core/mapper.py`
- Create: `src/asciart/core/engine.py`
- Create: `tests/test_engine.py`
- Create: `tests/conftest.py`

**Step 1: Write test fixtures**

```python
# tests/conftest.py
import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def white_image():
    """A 100x100 pure white image."""
    return Image.new("RGB", (100, 100), (255, 255, 255))


@pytest.fixture
def black_image():
    """A 100x100 pure black image."""
    return Image.new("RGB", (100, 100), (0, 0, 0))


@pytest.fixture
def gradient_image():
    """A 256x1 horizontal gradient from black to white."""
    img = Image.new("RGB", (256, 1))
    for x in range(256):
        img.putpixel((x, 0), (x, x, x))
    return img


@pytest.fixture
def small_color_image():
    """A 4x4 image with distinct colors."""
    img = Image.new("RGB", (4, 4))
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (128, 128, 128), (0, 0, 0), (255, 255, 255), (128, 0, 0),
        (0, 128, 0), (0, 0, 128), (64, 64, 64), (192, 192, 192),
        (255, 128, 0), (128, 0, 255), (0, 255, 128), (64, 128, 192),
    ]
    for i, color in enumerate(colors):
        img.putpixel((i % 4, i // 4), color)
    return img
```

**Step 2: Write failing engine tests**

```python
# tests/test_engine.py
from PIL import Image
from asciart.core.engine import convert
from asciart.models import ConvertOptions, Mode, ColorMode, AsciiArt


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
    # White = highest luminance = last char in ramp (when not inverted)
    # But convention: last char in ramp is darkest. White should map to first char (space).
    # Our ramp " .:#" goes light-to-dark, so white -> " "
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
    # A full gradient should use most/all chars in the ramp
    assert len(chars_used) >= 8


def test_color_mode_attaches_rgb(small_color_image):
    opts = ConvertOptions(width=4, color=ColorMode.TRUECOLOR)
    result = convert(small_color_image, opts)
    # Every cell should have an fg color tuple
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
    """Height should be roughly width * (img_h/img_w) * font_ratio."""
    opts = ConvertOptions(width=20, font_ratio=0.5)
    result = convert(white_image, opts)
    # 100x100 image at width=20, ratio 0.5 -> height ~ 20 * 1.0 * 0.5 = 10
    assert result.height == 10
```

```bash
uv run pytest tests/test_engine.py -v
```

Expected: FAIL — `engine` module doesn't exist yet.

**Step 3: Write preprocess.py**

```python
# src/asciart/core/preprocess.py
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance


def preprocess(image: Image.Image, brightness: float, contrast: float, saturation: float, sharpness: float) -> Image.Image:
    """Apply image adjustments. All values are multipliers (1.0 = no change) except brightness which is an offset."""
    img = image.convert("RGB")

    if brightness != 0.0:
        enhancer = ImageEnhance.Brightness(img)
        # Map offset (-100..100) to multiplier (0.0..2.0)
        img = enhancer.enhance(1.0 + brightness / 100.0)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)

    return img


def resize(image: Image.Image, width: int, height: int | None, font_ratio: float) -> Image.Image:
    """Resize image to target character dimensions with aspect ratio correction."""
    img_w, img_h = image.size
    if height is None:
        height = int((img_h / img_w) * width * font_ratio)
        height = max(1, height)
    return image.resize((width, height), Image.LANCZOS)


def to_grayscale(pixels: np.ndarray) -> np.ndarray:
    """Convert RGB pixel array to grayscale using BT.709 weights. Returns float64 array [0, 255]."""
    return pixels[:, :, 0] * 0.2126 + pixels[:, :, 1] * 0.7152 + pixels[:, :, 2] * 0.0722
```

**Step 4: Write mapper.py**

```python
# src/asciart/core/mapper.py
from __future__ import annotations

import numpy as np
from asciart.models import Cell


def map_brightness(gray: np.ndarray, chars: str, invert: bool, colors: np.ndarray | None = None) -> list[list[Cell]]:
    """Map grayscale values to characters from a ramp.

    Args:
        gray: 2D float array (H, W) with values 0-255.
        chars: Character ramp string, ordered light-to-dark.
        invert: If True, reverse the ramp.
        colors: Optional RGB array (H, W, 3) for color attachment.

    Returns:
        2D list of Cells.
    """
    ramp = chars[::-1] if invert else chars
    num_chars = len(ramp)

    # Vectorized index computation
    indices = np.clip((gray / 255.0 * (num_chars - 1)).astype(int), 0, num_chars - 1)

    rows = []
    h, w = gray.shape
    for y in range(h):
        row = []
        for x in range(w):
            char = ramp[indices[y, x]]
            fg = None
            if colors is not None:
                fg = (int(colors[y, x, 0]), int(colors[y, x, 1]), int(colors[y, x, 2]))
            row.append(Cell(char=char, fg=fg))
        rows.append(row)

    return rows
```

**Step 5: Write engine.py**

```python
# src/asciart/core/engine.py
from __future__ import annotations

import numpy as np
from PIL import Image

from asciart.models import AsciiArt, ConvertOptions, ColorMode, Mode
from asciart.core.preprocess import preprocess, resize, to_grayscale
from asciart.core.mapper import map_brightness
from asciart.core.ramps import BLOCKS


def convert(image: Image.Image, options: ConvertOptions) -> AsciiArt:
    """Main conversion pipeline: Image + Options -> AsciiArt grid."""
    # 1. Preprocess
    img = preprocess(image, options.brightness, options.contrast, options.saturation, options.sharpness)

    # 2. Resize with aspect ratio correction
    img = resize(img, options.width, options.height, options.font_ratio)

    # 3. Convert to numpy
    pixels = np.array(img, dtype=np.float64)

    # 4. Grayscale
    gray = to_grayscale(pixels)

    # 5. Determine if we need color
    need_color = options.color not in (ColorMode.NONE,)
    if options.color == ColorMode.AUTO:
        need_color = True
    colors = pixels if need_color else None

    # 6. Pick character ramp based on mode
    if options.mode == Mode.BLOCKS:
        chars = BLOCKS
    else:
        chars = options.chars

    # 7. Map to characters
    cells = map_brightness(gray, chars, options.invert, colors)

    h, w = gray.shape
    return AsciiArt(width=w, height=h, cells=cells)
```

**Step 6: Run tests**

```bash
uv run pytest tests/test_engine.py -v
```

Expected: all 8 tests PASS.

**Step 7: Commit**

```bash
git add .
git commit -m "feat: core engine with brightness mapping, preprocessing, and aspect correction"
```

---

## Task 3: Terminal Renderer + Basic CLI

**Files:**
- Create: `src/asciart/renderers/terminal.py`
- Create: `src/asciart/renderers/text.py`
- Create: `src/asciart/cli.py`
- Create: `tests/test_renderers.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing renderer tests**

```python
# tests/test_renderers.py
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
```

```bash
uv run pytest tests/test_renderers.py -v
```

Expected: FAIL — modules don't exist.

**Step 2: Write text renderer**

```python
# src/asciart/renderers/text.py
from __future__ import annotations

from asciart.models import AsciiArt


def render_text(art: AsciiArt) -> str:
    """Render AsciiArt as plain text with no color codes."""
    lines = []
    for row in art.cells:
        lines.append("".join(cell.char for cell in row))
    return "\n".join(lines)
```

**Step 3: Write terminal renderer**

```python
# src/asciart/renderers/terminal.py
from __future__ import annotations

from asciart.models import AsciiArt

RESET = "\033[0m"


def _truecolor_fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


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
            if cell.fg is not None:
                if use_256:
                    parts.append(f"{_ansi256_fg(*cell.fg)}{cell.char}")
                else:
                    parts.append(f"{_truecolor_fg(*cell.fg)}{cell.char}")
            else:
                parts.append(cell.char)
        lines.append("".join(parts) + RESET)
    return "\n".join(lines)
```

**Step 4: Run renderer tests**

```bash
uv run pytest tests/test_renderers.py -v
```

Expected: all 6 tests PASS.

**Step 5: Write CLI**

```python
# src/asciart/cli.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from PIL import Image

from asciart.models import ConvertOptions, Mode, ColorMode, DitherMode, OutputFormat
from asciart.core.engine import convert
from asciart.renderers.terminal import render_ansi
from asciart.renderers.text import render_text

app = typer.Typer(name="asciart", help="Convert images to stunning ASCII art.")


@app.command()
def convert_cmd(
    image_path: Path = typer.Argument(..., help="Path to the image file", exists=True),
    width: int = typer.Option(80, "-w", "--width", help="Output width in characters"),
    mode: Mode = typer.Option(Mode.ASCII, "--mode", help="Rendering mode"),
    color: ColorMode = typer.Option(ColorMode.AUTO, "--color", help="Color mode"),
    dither: DitherMode = typer.Option(DitherMode.NONE, "--dither", help="Dithering algorithm"),
    invert: bool = typer.Option(False, "-i", "--invert", help="Invert brightness"),
    brightness: float = typer.Option(0.0, "--brightness", help="Brightness offset (-100 to 100)"),
    contrast: float = typer.Option(1.0, "--contrast", help="Contrast multiplier"),
    font_ratio: float = typer.Option(0.5, "--ratio", help="Font cell width/height ratio"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Use a preset"),
    format: OutputFormat = typer.Option(OutputFormat.ANSI, "--format", help="Output format"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save to file"),
):
    """Convert an image to ASCII art."""
    img = Image.open(image_path)

    options = ConvertOptions(
        width=width,
        mode=mode,
        color=color,
        dither=dither,
        invert=invert,
        brightness=brightness,
        contrast=contrast,
        font_ratio=font_ratio,
    )

    art = convert(img, options)

    # Render based on format
    if format == OutputFormat.TEXT:
        rendered = render_text(art)
    elif format == OutputFormat.ANSI:
        use_256 = color == ColorMode.ANSI256
        rendered = render_ansi(art, use_256=use_256)
    else:
        typer.echo(f"Format '{format.value}' not yet implemented.", err=True)
        raise typer.Exit(code=1)

    if output:
        output.write_text(rendered)
        typer.echo(f"Saved to {output}")
    else:
        typer.echo(rendered)


@app.command()
def play(
    image_path: Path = typer.Argument(..., help="Path to the image file", exists=True),
):
    """Open interactive mode with live preview."""
    typer.echo("Interactive mode not yet implemented.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
```

**Step 6: Write CLI smoke test**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from asciart.cli import app
from PIL import Image
import tempfile
from pathlib import Path

runner = CliRunner()


def _create_test_image(path: Path):
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    img.save(str(path))


def test_convert_basic():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(Path(f.name))
        result = runner.invoke(app, ["convert-cmd", f.name, "-w", "20", "--color", "none"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) > 0
        assert len(lines[0]) == 20


def test_convert_save_to_file():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_f:
        _create_test_image(Path(img_f.name))
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out_f:
            result = runner.invoke(
                app, ["convert-cmd", img_f.name, "-w", "20", "--format", "text", "-o", out_f.name]
            )
            assert result.exit_code == 0
            content = Path(out_f.name).read_text()
            assert len(content) > 0
```

**Step 7: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 8: Manual smoke test**

Download or use any image and run:

```bash
uv run asciart convert-cmd path/to/image.jpg -w 60 --color none
```

Expected: ASCII art prints to terminal.

**Step 9: Commit**

```bash
git add .
git commit -m "feat: terminal/text renderers and Typer CLI with convert command"
```

---

## Task 4: Braille, Block, and Half-Block Modes

**Files:**
- Modify: `src/asciart/core/mapper.py` — add `map_braille`, `map_halfblock`
- Modify: `src/asciart/core/engine.py` — route modes to correct mapper
- Create: `tests/test_braille.py`

**Step 1: Write failing braille/block tests**

```python
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
            # Braille chars are in range U+2800 - U+28FF
            assert 0x2800 <= ord(cell.char) <= 0x28FF


def test_braille_white_is_empty():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    opts = ConvertOptions(width=20, mode=Mode.BRAILLE, color=ColorMode.NONE)
    result = convert(img, opts)
    for row in result.cells:
        for cell in row:
            # White -> all dots off -> U+2800 (empty braille)
            assert cell.char == "\u2800"


def test_braille_black_is_full():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    opts = ConvertOptions(width=20, mode=Mode.BRAILLE, color=ColorMode.NONE)
    result = convert(img, opts)
    for row in result.cells:
        for cell in row:
            # Black -> all dots on -> U+28FF (full braille)
            assert cell.char == "\u28FF"


def test_blocks_mode_uses_block_chars():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    opts = ConvertOptions(width=20, mode=Mode.BLOCKS, color=ColorMode.NONE)
    result = convert(img, opts)
    block_chars = set(" ░▒▓█")
    for row in result.cells:
        for cell in row:
            assert cell.char in block_chars


def test_halfblock_mode_has_fg_and_bg():
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    opts = ConvertOptions(width=20, mode=Mode.HALFBLOCK, color=ColorMode.TRUECOLOR)
    result = convert(img, opts)
    for row in result.cells:
        for cell in row:
            assert cell.char == "▀"
            assert cell.fg is not None
            assert cell.bg is not None
```

```bash
uv run pytest tests/test_braille.py -v
```

Expected: FAIL — braille/halfblock mappers don't exist.

**Step 2: Add braille and halfblock mappers to mapper.py**

Add to `src/asciart/core/mapper.py`:

```python
def map_braille(gray: np.ndarray, threshold: float = 128.0, colors: np.ndarray | None = None) -> list[list[Cell]]:
    """Map grayscale image to braille characters.

    Each braille char encodes a 2x4 pixel block. The image dimensions should already
    account for this (width = output_cols * 2, height = output_rows * 4).
    """
    h, w = gray.shape
    # Pad to multiples of 4 (height) and 2 (width)
    pad_h = (4 - h % 4) % 4
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
        if colors is not None:
            colors = np.pad(colors, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)

    h, w = gray.shape
    rows_out = h // 4
    cols_out = w // 2

    # Braille dot positions: (row_offset, col_offset, bit_value)
    dot_map = [
        (0, 0, 0x01), (1, 0, 0x02), (2, 0, 0x04),
        (0, 1, 0x08), (1, 1, 0x10), (2, 1, 0x20),
        (3, 0, 0x40), (3, 1, 0x80),
    ]

    rows = []
    for by in range(rows_out):
        row = []
        for bx in range(cols_out):
            y0 = by * 4
            x0 = bx * 2
            offset = 0
            for dy, dx, bit in dot_map:
                if gray[y0 + dy, x0 + dx] < threshold:
                    offset |= bit

            fg = None
            if colors is not None:
                # Average color of the 2x4 block
                block = colors[y0:y0 + 4, x0:x0 + 2]
                avg = block.mean(axis=(0, 1))
                fg = (int(avg[0]), int(avg[1]), int(avg[2]))

            row.append(Cell(char=chr(0x2800 + offset), fg=fg))
        rows.append(row)

    return rows


def map_halfblock(pixels: np.ndarray) -> list[list[Cell]]:
    """Map pixels to half-block characters with fg/bg color.

    Each character cell encodes two vertically stacked pixels using ▀ (upper half block).
    FG = top pixel color, BG = bottom pixel color. Doubles vertical resolution.
    """
    h, w, _ = pixels.shape
    # Pad height to even
    if h % 2 != 0:
        pixels = np.pad(pixels, ((0, 1), (0, 0), (0, 0)), mode="edge")
        h += 1

    rows = []
    for y in range(0, h, 2):
        row = []
        for x in range(w):
            top = pixels[y, x]
            bottom = pixels[y + 1, x]
            fg = (int(top[0]), int(top[1]), int(top[2]))
            bg = (int(bottom[0]), int(bottom[1]), int(bottom[2]))
            row.append(Cell(char="▀", fg=fg, bg=bg))
        rows.append(row)

    return rows
```

**Step 3: Update engine.py to route modes**

Update `src/asciart/core/engine.py` `convert` function to handle braille and halfblock:

```python
# Replace the mapping section (step 7 onward) in engine.py:

from asciart.core.mapper import map_brightness, map_braille, map_halfblock

def convert(image: Image.Image, options: ConvertOptions) -> AsciiArt:
    """Main conversion pipeline: Image + Options -> AsciiArt grid."""
    # 1. Preprocess
    img = preprocess(image, options.brightness, options.contrast, options.saturation, options.sharpness)

    # 2. Determine if we need color
    need_color = options.color not in (ColorMode.NONE,)
    if options.color == ColorMode.AUTO:
        need_color = True

    # 3. Mode-specific sizing and mapping
    if options.mode == Mode.BRAILLE:
        # Braille: each char = 2x4 pixels, so resize to 2x width and 4x height
        target_w = options.width * 2
        img_w, img_h = img.size
        target_h = options.height
        if target_h is None:
            target_h = int((img_h / img_w) * target_w * options.font_ratio)
            # Round up to multiple of 4
            target_h = max(4, ((target_h + 3) // 4) * 4)

        img = img.resize((target_w, target_h), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        gray = to_grayscale(pixels)
        colors = pixels if need_color else None
        if options.invert:
            gray = 255.0 - gray
        cells = map_braille(gray, options.edge_threshold, colors)

    elif options.mode == Mode.HALFBLOCK:
        # Halfblock: each char = 1x2 pixels, so resize to width and 2x height
        target_w = options.width
        img_w, img_h = img.size
        target_h = options.height
        if target_h is None:
            target_h = int((img_h / img_w) * target_w * options.font_ratio * 2)
            target_h = max(2, target_h)
            # Round up to even
            if target_h % 2 != 0:
                target_h += 1

        img = img.resize((target_w, target_h), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        cells = map_halfblock(pixels)

    else:
        # ASCII and BLOCKS modes
        img = resize(img, options.width, options.height, options.font_ratio)
        pixels = np.array(img, dtype=np.float64)
        gray = to_grayscale(pixels)
        colors = pixels if need_color else None

        if options.mode == Mode.BLOCKS:
            chars = BLOCKS
        else:
            chars = options.chars

        cells = map_brightness(gray, chars, options.invert, colors)

    out_h = len(cells)
    out_w = len(cells[0]) if cells else 0
    return AsciiArt(width=out_w, height=out_h, cells=cells)
```

**Step 4: Update terminal renderer for bg color (halfblock)**

Add bg color support in `src/asciart/renderers/terminal.py`:

```python
def _truecolor_bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"
```

And update `render_ansi` to handle bg:

```python
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
```

**Step 5: Run tests**

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 6: Commit**

```bash
git add .
git commit -m "feat: braille, block, and half-block rendering modes"
```

---

## Task 5: Dithering Algorithms

**Files:**
- Create: `src/asciart/core/dither.py`
- Modify: `src/asciart/core/engine.py` — apply dithering in pipeline
- Create: `tests/test_dither.py`

**Step 1: Write failing dither tests**

```python
# tests/test_dither.py
import numpy as np
from asciart.core.dither import floyd_steinberg, ordered_dither


def test_floyd_steinberg_preserves_shape():
    img = np.random.rand(10, 20) * 255
    result = floyd_steinberg(img, num_levels=4)
    assert result.shape == (10, 20)


def test_floyd_steinberg_quantizes_to_levels():
    img = np.random.rand(10, 20) * 255
    result = floyd_steinberg(img, num_levels=4)
    unique = np.unique(np.round(result).astype(int))
    # Should only have values corresponding to 4 levels: 0, 85, 170, 255
    assert len(unique) <= 4


def test_floyd_steinberg_uniform_stays_uniform():
    img = np.full((10, 10), 128.0)
    result = floyd_steinberg(img, num_levels=10)
    # Uniform input should produce mostly uniform output
    assert np.std(result) < 30


def test_ordered_dither_preserves_shape():
    img = np.random.rand(10, 20) * 255
    result = ordered_dither(img, num_levels=4)
    assert result.shape == (10, 20)


def test_ordered_dither_quantizes():
    img = np.full((16, 16), 128.0)
    result = ordered_dither(img, num_levels=4)
    unique = np.unique(np.round(result).astype(int))
    assert len(unique) <= 4


def test_dither_gradient_uses_more_values_than_naive():
    """A gradient dithered should have smoother transitions than naive quantization."""
    gradient = np.tile(np.linspace(0, 255, 100), (10, 1))
    dithered = floyd_steinberg(gradient, num_levels=4)
    # Check that dithering happened (output differs from simple quantization)
    naive = np.round(gradient / 255 * 3) / 3 * 255
    assert not np.array_equal(dithered, naive)
```

```bash
uv run pytest tests/test_dither.py -v
```

Expected: FAIL.

**Step 2: Implement dither.py**

```python
# src/asciart/core/dither.py
from __future__ import annotations

import numpy as np


def floyd_steinberg(image: np.ndarray, num_levels: int) -> np.ndarray:
    """Apply Floyd-Steinberg error diffusion dithering.

    Args:
        image: 2D float array [0, 255].
        num_levels: Number of quantization levels (matches character ramp length).

    Returns:
        Dithered 2D float array with values snapped to `num_levels` evenly spaced values.
    """
    img = image.astype(np.float64).copy()
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            old = img[y, x]
            # Quantize to nearest level
            level = round(old / 255.0 * (num_levels - 1))
            level = max(0, min(num_levels - 1, level))
            new = level / (num_levels - 1) * 255.0
            img[y, x] = new
            error = old - new

            if x + 1 < w:
                img[y, x + 1] += error * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += error * 3 / 16
                img[y + 1, x] += error * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 1 / 16

    return np.clip(img, 0, 255)


# 4x4 Bayer matrix for ordered dithering
BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float64) / 16.0


def ordered_dither(image: np.ndarray, num_levels: int) -> np.ndarray:
    """Apply ordered (Bayer matrix) dithering.

    Args:
        image: 2D float array [0, 255].
        num_levels: Number of quantization levels.

    Returns:
        Dithered 2D float array.
    """
    h, w = image.shape
    # Tile the Bayer matrix to cover the image
    mh, mw = BAYER_4X4.shape
    threshold = np.tile(BAYER_4X4, ((h + mh - 1) // mh, (w + mw - 1) // mw))[:h, :w]

    normalized = image / 255.0
    # Add threshold bias and quantize
    biased = normalized + (threshold - 0.5) / num_levels
    quantized = np.round(biased * (num_levels - 1)).clip(0, num_levels - 1)
    return quantized / (num_levels - 1) * 255.0
```

**Step 3: Wire dithering into engine.py**

In the ASCII/BLOCKS branch of `convert()`, after computing `gray` and before calling `map_brightness`, add:

```python
from asciart.core.dither import floyd_steinberg, ordered_dither
from asciart.models import DitherMode

# Add inside the ASCII/BLOCKS branch, after grayscale:
if options.dither == DitherMode.FLOYD_STEINBERG:
    num_levels = len(chars)
    gray = floyd_steinberg(gray, num_levels)
elif options.dither == DitherMode.ORDERED:
    num_levels = len(chars)
    gray = ordered_dither(gray, num_levels)
```

**Step 4: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add .
git commit -m "feat: Floyd-Steinberg and ordered dithering algorithms"
```

---

## Task 6: Edge Detection

**Files:**
- Create: `src/asciart/core/edges.py`
- Modify: `src/asciart/core/mapper.py` — add edge character support
- Modify: `src/asciart/core/engine.py` — apply edge detection in pipeline
- Create: `tests/test_edges.py`

**Step 1: Write failing edge tests**

```python
# tests/test_edges.py
import numpy as np
from asciart.core.edges import detect_edges


def test_detect_edges_returns_mask_and_angles():
    img = np.zeros((20, 20))
    img[:, 10:] = 255  # vertical edge in the middle
    mask, angles = detect_edges(img, threshold=50)
    assert mask.shape == (20, 20)
    assert angles.shape == (20, 20)
    assert mask.dtype == bool


def test_detects_vertical_edge():
    img = np.zeros((20, 20))
    img[:, 10:] = 255
    mask, angles = detect_edges(img, threshold=30)
    # Should detect edges near column 10
    assert mask[:, 9:12].any()


def test_uniform_image_no_edges():
    img = np.full((20, 20), 128.0)
    mask, angles = detect_edges(img, threshold=50)
    assert not mask.any()


def test_edge_chars_in_output():
    from PIL import Image
    from asciart.core.engine import convert
    from asciart.models import ConvertOptions, ColorMode

    # Create an image with a strong vertical edge
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        for y in range(100):
            if x < 50:
                img.putpixel((x, y), (0, 0, 0))
            else:
                img.putpixel((x, y), (255, 255, 255))

    opts = ConvertOptions(width=40, edge_detection=True, color=ColorMode.NONE)
    result = convert(img, opts)
    chars_used = {cell.char for row in result.cells for cell in row}
    # Should contain edge characters
    edge_chars = set("|-/\\")
    assert chars_used & edge_chars
```

```bash
uv run pytest tests/test_edges.py -v
```

Expected: FAIL.

**Step 2: Implement edges.py**

```python
# src/asciart/core/edges.py
from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def detect_edges(gray: np.ndarray, threshold: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Detect edges using Sobel filter.

    Args:
        gray: 2D float array [0, 255].
        threshold: Minimum gradient magnitude to consider an edge.

    Returns:
        Tuple of (edge_mask: bool array, angles: float array in radians).
    """
    gx = convolve2d(gray, SOBEL_X, mode="same", boundary="symm")
    gy = convolve2d(gray, SOBEL_Y, mode="same", boundary="symm")

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angles = np.arctan2(gy, gx)

    mask = magnitude > threshold
    return mask, angles


def angle_to_char(angle: float) -> str:
    """Map edge angle (radians) to a directional ASCII character."""
    a = angle % np.pi
    if a < np.pi / 8 or a > 7 * np.pi / 8:
        return "-"
    elif a < 3 * np.pi / 8:
        return "/"
    elif a < 5 * np.pi / 8:
        return "|"
    else:
        return "\\"
```

**Step 3: Add scipy as a dependency**

```bash
uv add scipy
```

**Step 4: Wire edges into engine.py**

In the ASCII/BLOCKS branch, after dithering and before `map_brightness`, add edge detection support. Update `map_brightness` call to accept edge data, or apply edges as a post-step. The cleaner approach is a post-step that overrides specific cells:

Add to `engine.py` in the ASCII/BLOCKS branch:

```python
from asciart.core.edges import detect_edges, angle_to_char

# After cells = map_brightness(...)
if options.edge_detection:
    edge_mask, edge_angles = detect_edges(gray, options.edge_threshold)
    for y in range(len(cells)):
        for x in range(len(cells[y])):
            if edge_mask[y, x]:
                cells[y][x] = Cell(char=angle_to_char(edge_angles[y, x]), fg=cells[y][x].fg)
```

**Step 5: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 6: Commit**

```bash
git add .
git commit -m "feat: Sobel edge detection with directional ASCII characters"
```

---

## Task 7: HTML, PNG, and SVG Renderers

**Files:**
- Create: `src/asciart/renderers/html.py`
- Create: `src/asciart/renderers/image.py`
- Create: `src/asciart/renderers/svg.py`
- Modify: `src/asciart/cli.py` — wire new formats
- Create: `tests/test_export.py`

**Step 1: Write failing export tests**

```python
# tests/test_export.py
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
    # Should have inline color styles
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
    # PNG magic bytes
    assert data[:4] == b"\x89PNG"


def test_png_nonzero_size():
    art = _make_colored_art()
    data = render_png(art)
    assert len(data) > 100
```

```bash
uv run pytest tests/test_export.py -v
```

Expected: FAIL.

**Step 2: Implement HTML renderer**

```python
# src/asciart/renderers/html.py
from __future__ import annotations

from html import escape
from asciart.models import AsciiArt


def render_html(art: AsciiArt, title: str = "ASCII Art") -> str:
    """Render AsciiArt as a self-contained HTML file."""
    lines = []
    for row in art.cells:
        parts = []
        for cell in row:
            char = escape(cell.char)
            if char == " ":
                char = "&nbsp;"
            if cell.fg:
                r, g, b = cell.fg
                style = f"color:rgb({r},{g},{b})"
                if cell.bg:
                    br, bg, bb = cell.bg
                    style += f";background:rgb({br},{bg},{bb})"
                parts.append(f'<span style="{style}">{char}</span>')
            else:
                parts.append(char)
        lines.append("".join(parts))

    body = "\n".join(lines)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
  body {{ background: #1e1e1e; display: flex; justify-content: center; padding: 20px; }}
  pre {{ font-family: 'Courier New', monospace; font-size: 10px; line-height: 1.0; color: #ccc; }}
</style>
</head>
<body>
<pre>{body}</pre>
</body>
</html>"""
```

**Step 3: Implement SVG renderer**

```python
# src/asciart/renderers/svg.py
from __future__ import annotations

from html import escape
from asciart.models import AsciiArt

CHAR_WIDTH = 7.2
CHAR_HEIGHT = 14


def render_svg(art: AsciiArt) -> str:
    """Render AsciiArt as an SVG document."""
    svg_w = art.width * CHAR_WIDTH + 20
    svg_h = art.height * CHAR_HEIGHT + 20

    text_elements = []
    for y, row in enumerate(art.cells):
        # Group consecutive same-color characters for efficiency
        x_pos = 10
        y_pos = 10 + (y + 1) * CHAR_HEIGHT
        line_chars = ""
        current_color = None

        for cell in row:
            color = f"rgb({cell.fg[0]},{cell.fg[1]},{cell.fg[2]})" if cell.fg else "#cccccc"
            if color != current_color:
                if line_chars:
                    text_elements.append(
                        f'<text x="{x_pos}" y="{y_pos}" fill="{current_color}">'
                        f"{escape(line_chars)}</text>"
                    )
                    x_pos += len(line_chars) * CHAR_WIDTH
                line_chars = cell.char
                current_color = color
            else:
                line_chars += cell.char

        if line_chars:
            text_elements.append(
                f'<text x="{x_pos}" y="{y_pos}" fill="{current_color}">'
                f"{escape(line_chars)}</text>"
            )

    elements = "\n  ".join(text_elements)

    return f"""<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" style="background:#1e1e1e">
  <style>text {{ font-family: 'Courier New', monospace; font-size: 12px; }}</style>
  {elements}
</svg>"""
```

**Step 4: Implement PNG renderer**

```python
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
    """Render AsciiArt as a PNG image. Returns PNG bytes."""
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
```

**Step 5: Wire into CLI**

Update `src/asciart/cli.py` `convert_cmd` to handle all formats:

```python
from asciart.renderers.html import render_html
from asciart.renderers.svg import render_svg
from asciart.renderers.image import render_png

# Replace the format routing section:
if format == OutputFormat.TEXT:
    rendered = render_text(art)
elif format == OutputFormat.ANSI:
    use_256 = color == ColorMode.ANSI256
    rendered = render_ansi(art, use_256=use_256)
elif format == OutputFormat.HTML:
    rendered = render_html(art)
elif format == OutputFormat.SVG:
    rendered = render_svg(art)
elif format == OutputFormat.PNG:
    if not output:
        typer.echo("PNG format requires --output/-o flag.", err=True)
        raise typer.Exit(code=1)
    data = render_png(art)
    output.write_bytes(data)
    typer.echo(f"Saved PNG to {output}")
    return

if output:
    output.write_text(rendered)
    typer.echo(f"Saved to {output}")
else:
    typer.echo(rendered)
```

**Step 6: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 7: Commit**

```bash
git add .
git commit -m "feat: HTML, PNG, and SVG export renderers"
```

---

## Task 8: Preset System

**Files:**
- Create: `src/asciart/presets.py`
- Modify: `src/asciart/cli.py` — apply presets
- Create: `tests/test_presets.py`

**Step 1: Write failing preset tests**

```python
# tests/test_presets.py
from asciart.presets import PRESETS, get_preset
from asciart.models import ConvertOptions, Mode, DitherMode


def test_all_presets_are_valid():
    for name, opts in PRESETS.items():
        assert isinstance(opts, ConvertOptions)
        assert isinstance(name, str)


def test_photo_preset():
    opts = get_preset("photo")
    assert opts.dither == DitherMode.FLOYD_STEINBERG
    assert opts.contrast > 1.0


def test_retro_preset():
    opts = get_preset("retro")
    assert opts.chars == " .:#"
    assert opts.dither == DitherMode.ORDERED


def test_unknown_preset_returns_none():
    assert get_preset("nonexistent") is None


def test_presets_contain_expected_names():
    expected = {"photo", "logo", "retro", "hd", "blocks", "lineart"}
    assert expected.issubset(set(PRESETS.keys()))
```

**Step 2: Implement presets.py**

```python
# src/asciart/presets.py
from __future__ import annotations

from asciart.models import ConvertOptions, Mode, ColorMode, DitherMode

PRESETS: dict[str, ConvertOptions] = {
    "photo": ConvertOptions(
        mode=Mode.ASCII,
        chars=" .:-=+*#%@",
        color=ColorMode.TRUECOLOR,
        dither=DitherMode.FLOYD_STEINBERG,
        contrast=1.2,
    ),
    "logo": ConvertOptions(
        mode=Mode.BRAILLE,
        color=ColorMode.TRUECOLOR,
        contrast=1.5,
        edge_detection=True,
    ),
    "retro": ConvertOptions(
        mode=Mode.ASCII,
        chars=" .:#",
        color=ColorMode.NONE,
        dither=DitherMode.ORDERED,
    ),
    "hd": ConvertOptions(
        mode=Mode.BRAILLE,
        color=ColorMode.TRUECOLOR,
        width=200,
        dither=DitherMode.FLOYD_STEINBERG,
    ),
    "blocks": ConvertOptions(
        mode=Mode.HALFBLOCK,
        color=ColorMode.TRUECOLOR,
    ),
    "lineart": ConvertOptions(
        mode=Mode.ASCII,
        edge_detection=True,
        color=ColorMode.NONE,
        contrast=1.8,
    ),
}


def get_preset(name: str) -> ConvertOptions | None:
    """Get a preset by name, or None if not found."""
    return PRESETS.get(name)
```

**Step 3: Wire presets into CLI**

In `src/asciart/cli.py`, at the top of `convert_cmd`, after parsing args:

```python
from asciart.presets import get_preset
from dataclasses import replace

# Inside convert_cmd, before building ConvertOptions:
if preset:
    base = get_preset(preset)
    if base is None:
        typer.echo(f"Unknown preset '{preset}'. Available: photo, logo, retro, hd, blocks, lineart", err=True)
        raise typer.Exit(code=1)
    # Apply CLI overrides on top of preset
    options = replace(base, width=width, font_ratio=font_ratio)
    if invert:
        options = replace(options, invert=True)
else:
    options = ConvertOptions(
        width=width,
        mode=mode,
        color=color,
        dither=dither,
        invert=invert,
        brightness=brightness,
        contrast=contrast,
        font_ratio=font_ratio,
    )
```

**Step 4: Run all tests**

```bash
uv run pytest -v
```

**Step 5: Commit**

```bash
git add .
git commit -m "feat: preset system with photo, logo, retro, hd, blocks, lineart"
```

---

## Task 9: Interactive TUI with Live Preview

This is the largest task. Build the Textual app with sliders, dropdowns, and live preview.

**Files:**
- Create: `src/asciart/tui.py`
- Create: `src/asciart/tui.tcss`
- Modify: `src/asciart/cli.py` — wire `play` command
- Create: `tests/test_tui.py`

**Step 1: Write basic TUI smoke test**

```python
# tests/test_tui.py
import pytest
from asciart.tui import AsciiArtApp


@pytest.mark.asyncio
async def test_tui_app_creates():
    """Smoke test: the app can be instantiated."""
    app = AsciiArtApp(image_path="nonexistent.png")
    assert app is not None
```

Add pytest-asyncio:

```bash
uv add --dev pytest-asyncio
```

**Step 2: Write the TUI CSS**

```css
/* src/asciart/tui.tcss */
Screen {
    layout: horizontal;
}

#preview-container {
    width: 3fr;
    height: 100%;
    border: solid $primary;
}

#preview {
    width: 100%;
    height: 100%;
    overflow-y: auto;
    overflow-x: auto;
}

#controls {
    width: 1fr;
    min-width: 35;
    max-width: 45;
    height: 100%;
    overflow-y: auto;
    padding: 1;
    border: solid $secondary;
}

.section-label {
    text-style: bold;
    color: $text;
    margin-top: 1;
    margin-bottom: 0;
}

.control-row {
    height: auto;
    margin-bottom: 0;
}

.control-label {
    width: 100%;
    height: 1;
    color: $text-muted;
}

Slider {
    margin: 0 1;
}

#status-bar {
    dock: bottom;
    height: 1;
    background: $surface;
    color: $text-muted;
    padding: 0 1;
}

#preset-bar {
    height: 3;
    layout: horizontal;
    padding: 0 1;
}

#preset-bar Button {
    min-width: 10;
    margin: 0 1;
}
```

**Step 3: Write the TUI app**

```python
# src/asciart/tui.py
from __future__ import annotations

import time
from pathlib import Path

from PIL import Image
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Label, Select, Slider, Static, Switch
from textual.timer import Timer

from asciart.core.engine import convert
from asciart.models import ColorMode, ConvertOptions, DitherMode, Mode
from asciart.presets import PRESETS
from asciart.renderers.terminal import render_ansi
from asciart.renderers.text import render_text
from asciart.renderers.html import render_html
from asciart.renderers.image import render_png


class AsciiArtApp(App):
    CSS_PATH = "tui.tcss"
    TITLE = "asciart - interactive mode"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, image_path: str, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self._image: Image.Image | None = None
        self._debounce_timer: Timer | None = None
        self._options = ConvertOptions(width=80)

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal():
            # Left: Preview
            with VerticalScroll(id="preview-container"):
                yield Static("Loading...", id="preview")

            # Right: Controls
            with VerticalScroll(id="controls"):
                # Presets
                yield Label("Presets", classes="section-label")
                with Horizontal(id="preset-bar"):
                    for name in PRESETS:
                        yield Button(name.capitalize(), id=f"preset-{name}", variant="default")

                # Mode
                yield Label("Mode", classes="section-label")
                yield Select(
                    [(m.value, m.value) for m in Mode],
                    value=Mode.ASCII.value,
                    id="mode-select",
                )

                # Dimensions
                yield Label("Width", classes="section-label")
                yield Slider(min=20, max=300, value=80, step=5, id="width-slider")

                yield Label("Font Ratio", classes="section-label")
                yield Slider(min=20, max=100, value=50, step=5, id="ratio-slider")

                # Image adjustments
                yield Label("--- Image ---", classes="section-label")

                yield Label("Brightness", classes="control-label")
                yield Slider(min=-100, max=100, value=0, step=5, id="brightness-slider")

                yield Label("Contrast", classes="control-label")
                yield Slider(min=10, max=300, value=100, step=5, id="contrast-slider")

                yield Label("Saturation", classes="control-label")
                yield Slider(min=0, max=300, value=100, step=5, id="saturation-slider")

                yield Label("Sharpness", classes="control-label")
                yield Slider(min=0, max=300, value=100, step=5, id="sharpness-slider")

                # Rendering
                yield Label("--- Rendering ---", classes="section-label")

                yield Label("Dither", classes="section-label")
                yield Select(
                    [(d.value, d.value) for d in DitherMode],
                    value=DitherMode.NONE.value,
                    id="dither-select",
                )

                yield Label("Color", classes="section-label")
                yield Select(
                    [(c.value, c.value) for c in ColorMode],
                    value=ColorMode.TRUECOLOR.value,
                    id="color-select",
                )

                with Horizontal(classes="control-row"):
                    yield Label("Edge Detection")
                    yield Switch(id="edge-switch", value=False)

                with Horizontal(classes="control-row"):
                    yield Label("Invert")
                    yield Switch(id="invert-switch", value=False)

                # Export
                yield Label("--- Export ---", classes="section-label")
                with Horizontal(classes="control-row"):
                    yield Button("Save TXT", id="save-txt")
                    yield Button("Save HTML", id="save-html")
                with Horizontal(classes="control-row"):
                    yield Button("Save PNG", id="save-png")

        yield Static("Loading...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        try:
            self._image = Image.open(self.image_path)
        except Exception as e:
            self.query_one("#preview", Static).update(f"Error loading image: {e}")
            return
        self._refresh_preview()

    def _build_options(self) -> ConvertOptions:
        return ConvertOptions(
            width=self.query_one("#width-slider", Slider).value,
            mode=Mode(self.query_one("#mode-select", Select).value),
            color=ColorMode(self.query_one("#color-select", Select).value),
            dither=DitherMode(self.query_one("#dither-select", Select).value),
            invert=self.query_one("#invert-switch", Switch).value,
            brightness=float(self.query_one("#brightness-slider", Slider).value),
            contrast=self.query_one("#contrast-slider", Slider).value / 100.0,
            saturation=self.query_one("#saturation-slider", Slider).value / 100.0,
            sharpness=self.query_one("#sharpness-slider", Slider).value / 100.0,
            font_ratio=self.query_one("#ratio-slider", Slider).value / 100.0,
            edge_detection=self.query_one("#edge-switch", Switch).value,
        )

    def _refresh_preview(self) -> None:
        if self._image is None:
            return
        start = time.perf_counter()
        self._options = self._build_options()
        art = convert(self._image, self._options)
        rendered = render_text(art)
        elapsed = (time.perf_counter() - start) * 1000
        self.query_one("#preview", Static).update(rendered)
        status = f"{art.width}x{art.height} chars | {self._options.mode.value} | {self._options.color.value} | {elapsed:.0f}ms"
        self.query_one("#status-bar", Static).update(status)

    def _schedule_refresh(self) -> None:
        """Debounced refresh: wait 100ms after last change."""
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        self._debounce_timer = self.set_timer(0.1, self._refresh_preview)

    def on_slider_changed(self, event: Slider.Changed) -> None:
        self._schedule_refresh()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._schedule_refresh()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._schedule_refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        # Preset buttons
        if btn_id.startswith("preset-"):
            preset_name = btn_id.replace("preset-", "")
            preset = PRESETS.get(preset_name)
            if preset:
                self._apply_preset(preset)
            return

        # Export buttons
        if btn_id == "save-txt":
            self._save_export("txt")
        elif btn_id == "save-html":
            self._save_export("html")
        elif btn_id == "save-png":
            self._save_export("png")

    def _apply_preset(self, preset: ConvertOptions) -> None:
        """Snap all sliders/selects to preset values."""
        self.query_one("#mode-select", Select).value = preset.mode.value
        self.query_one("#width-slider", Slider).value = preset.width
        self.query_one("#brightness-slider", Slider).value = int(preset.brightness)
        self.query_one("#contrast-slider", Slider).value = int(preset.contrast * 100)
        self.query_one("#saturation-slider", Slider).value = int(preset.saturation * 100)
        self.query_one("#sharpness-slider", Slider).value = int(preset.sharpness * 100)
        self.query_one("#ratio-slider", Slider).value = int(preset.font_ratio * 100)
        self.query_one("#dither-select", Select).value = preset.dither.value
        self.query_one("#color-select", Select).value = preset.color.value
        self.query_one("#edge-switch", Switch).value = preset.edge_detection
        self.query_one("#invert-switch", Switch).value = preset.invert
        self._schedule_refresh()

    def _save_export(self, fmt: str) -> None:
        if self._image is None:
            return
        art = convert(self._image, self._options)
        stem = Path(self.image_path).stem

        if fmt == "txt":
            path = f"{stem}_ascii.txt"
            Path(path).write_text(render_text(art))
        elif fmt == "html":
            path = f"{stem}_ascii.html"
            Path(path).write_text(render_html(art))
        elif fmt == "png":
            path = f"{stem}_ascii.png"
            Path(path).write_bytes(render_png(art))
        else:
            return

        self.query_one("#status-bar", Static).update(f"Saved to {path}")
```

**Step 4: Wire `play` command in CLI**

Update `src/asciart/cli.py`:

```python
@app.command()
def play(
    image_path: Path = typer.Argument(..., help="Path to the image file", exists=True),
):
    """Open interactive mode with live preview."""
    from asciart.tui import AsciiArtApp
    tui = AsciiArtApp(image_path=str(image_path))
    tui.run()
```

**Step 5: Run all tests**

```bash
uv run pytest -v
```

**Step 6: Manual test**

```bash
uv run asciart play path/to/image.jpg
```

Expected: TUI opens with live preview, sliders work, presets snap values.

**Step 7: Commit**

```bash
git add .
git commit -m "feat: interactive TUI with live preview, sliders, presets, and export"
```

---

## Task 10: GIF Animation Support

**Files:**
- Create: `src/asciart/core/gif.py`
- Modify: `src/asciart/cli.py` — add `--animate` flag
- Create: `tests/test_gif.py`

**Step 1: Write failing GIF tests**

```python
# tests/test_gif.py
import tempfile
from pathlib import Path
from PIL import Image
from asciart.core.gif import extract_frames, convert_gif
from asciart.models import ConvertOptions, ColorMode


def _make_test_gif() -> Path:
    """Create a simple 3-frame animated GIF."""
    frames = []
    for i in range(3):
        img = Image.new("RGB", (50, 50), (i * 80, i * 80, i * 80))
        frames.append(img)

    path = Path(tempfile.mktemp(suffix=".gif"))
    frames[0].save(
        str(path), save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return path


def test_extract_frames():
    path = _make_test_gif()
    frames, durations = extract_frames(str(path))
    assert len(frames) == 3
    assert len(durations) == 3
    assert all(isinstance(f, Image.Image) for f in frames)


def test_convert_gif_returns_list():
    path = _make_test_gif()
    opts = ConvertOptions(width=10, color=ColorMode.NONE)
    arts = convert_gif(str(path), opts)
    assert len(arts) == 3
    assert all(a.width == 10 for a in arts)
```

**Step 2: Implement gif.py**

```python
# src/asciart/core/gif.py
from __future__ import annotations

from PIL import Image

from asciart.core.engine import convert
from asciart.models import AsciiArt, ConvertOptions


def extract_frames(path: str) -> tuple[list[Image.Image], list[int]]:
    """Extract all frames and durations from an animated GIF.

    Returns:
        Tuple of (frames as RGB PIL Images, durations in ms).
    """
    img = Image.open(path)
    frames = []
    durations = []

    try:
        while True:
            frames.append(img.convert("RGB").copy())
            durations.append(img.info.get("duration", 100))
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    return frames, durations


def convert_gif(path: str, options: ConvertOptions) -> list[AsciiArt]:
    """Convert all frames of an animated GIF to AsciiArt."""
    frames, _ = extract_frames(path)
    return [convert(frame, options) for frame in frames]
```

**Step 3: Add `--animate` to CLI**

In `src/asciart/cli.py`, add an `animate` flag to `convert_cmd` and playback logic:

```python
import sys
import time

# Add to convert_cmd params:
# animate: bool = typer.Option(False, "--animate", help="Animate GIF in terminal"),

# Add after art = convert(...):
# Handle GIF animation
if animate and str(image_path).lower().endswith(".gif"):
    from asciart.core.gif import extract_frames
    frames_pil, durations = extract_frames(str(image_path))
    arts = [convert(f, options) for f in frames_pil]

    # Hide cursor
    sys.stdout.write("\033[?25l")
    try:
        while True:
            for art_frame, duration in zip(arts, durations):
                if format == OutputFormat.ANSI:
                    rendered = render_ansi(art_frame, use_256=(color == ColorMode.ANSI256))
                else:
                    rendered = render_text(art_frame)
                sys.stdout.write("\033[H")  # cursor home
                sys.stdout.write(rendered)
                sys.stdout.flush()
                time.sleep(duration / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")  # show cursor
        sys.stdout.write("\n")
    return
```

**Step 4: Run all tests**

```bash
uv run pytest -v
```

**Step 5: Commit**

```bash
git add .
git commit -m "feat: GIF animation support with terminal playback"
```

---

## Task 11: Polish — Clipboard, Error Handling, Help Text

**Files:**
- Modify: `src/asciart/cli.py` — add `--copy`, better error messages, help text
- Create: `src/asciart/clipboard.py`
- Create: `tests/test_clipboard.py`

**Step 1: Implement clipboard.py**

```python
# src/asciart/clipboard.py
from __future__ import annotations

import subprocess
import sys


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    try:
        if sys.platform == "darwin":
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode())
            return proc.returncode == 0
        elif sys.platform.startswith("linux"):
            proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
            proc.communicate(text.encode())
            return proc.returncode == 0
        elif sys.platform == "win32":
            proc = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
            proc.communicate(text.encode())
            return proc.returncode == 0
    except FileNotFoundError:
        return False
    return False
```

**Step 2: Wire clipboard and add error handling to CLI**

Add to `convert_cmd` params:

```python
copy: bool = typer.Option(False, "--copy", help="Copy output to clipboard"),
```

After rendering:

```python
if copy:
    from asciart.clipboard import copy_to_clipboard
    plain = render_text(art)
    if copy_to_clipboard(plain):
        typer.echo("Copied to clipboard!", err=True)
    else:
        typer.echo("Failed to copy to clipboard (is xclip/pbcopy installed?)", err=True)
```

**Step 3: Run all tests**

```bash
uv run pytest -v
```

**Step 4: Final manual test of all features**

```bash
# Basic
uv run asciart convert-cmd test.jpg -w 80

# Colored
uv run asciart convert-cmd test.jpg -w 100 --color truecolor

# Braille
uv run asciart convert-cmd test.jpg -w 60 --mode braille

# Half-block
uv run asciart convert-cmd test.jpg -w 60 --mode halfblock --color truecolor

# Preset
uv run asciart convert-cmd test.jpg --preset photo

# Export
uv run asciart convert-cmd test.jpg --format html -o test.html
uv run asciart convert-cmd test.jpg --format png -o test.png

# Interactive
uv run asciart play test.jpg
```

**Step 5: Commit**

```bash
git add .
git commit -m "feat: clipboard support, error handling, polish"
```

---

## Summary

| Task | What It Builds | Key Tests |
|------|---------------|-----------|
| 1 | Project scaffolding, models, ramps | test_models.py |
| 2 | Core engine, brightness mapping | test_engine.py |
| 3 | Terminal/text renderers, CLI | test_renderers.py, test_cli.py |
| 4 | Braille, block, halfblock modes | test_braille.py |
| 5 | Floyd-Steinberg + ordered dithering | test_dither.py |
| 6 | Sobel edge detection | test_edges.py |
| 7 | HTML, PNG, SVG export | test_export.py |
| 8 | Preset system | test_presets.py |
| 9 | Interactive TUI with live preview | test_tui.py |
| 10 | GIF animation | test_gif.py |
| 11 | Clipboard, polish | test_clipboard.py |

Each task builds on the previous and is independently testable. The project goes from zero to fully functional in 11 commits.

# asciart — Image to ASCII Art Converter

## Overview

A Python CLI tool that converts images to high-quality ASCII art with multiple rendering modes, an interactive terminal UI for real-time tweaking, and multiple export formats. Designed with a clean core engine that can later power a web frontend.

## Architecture

```
asciart/
├── pyproject.toml
├── src/
│   └── asciart/
│       ├── __init__.py
│       ├── cli.py              # Typer CLI (one-shot mode)
│       ├── tui.py              # Textual app (interactive mode)
│       │
│       ├── core/               # Pure computation — ZERO I/O
│       │   ├── __init__.py
│       │   ├── engine.py       # Main pipeline orchestrator
│       │   ├── mapper.py       # Pixel -> character mapping
│       │   ├── dither.py       # Floyd-Steinberg, ordered dithering
│       │   ├── edges.py        # Sobel edge detection
│       │   ├── color.py        # Color quantization, ANSI codes
│       │   ├── preprocess.py   # Resize, contrast, brightness, sharpness
│       │   └── ramps.py        # Character ramp definitions & presets
│       │
│       ├── models.py           # Data structures (AsciiArt, Cell, Options)
│       │
│       └── renderers/          # AsciiArt grid -> output format
│           ├── __init__.py
│           ├── terminal.py     # ANSI colored terminal output
│           ├── html.py         # Standalone HTML with CSS colors
│           ├── text.py         # Plain .txt (no color)
│           ├── image.py        # Render to PNG via Pillow
│           └── svg.py          # SVG vector output
```

### Key Principle

`core/` has no imports from `cli`, `tui`, or `renderers`. It takes a NumPy array + an `Options` dataclass, returns an `AsciiArt` grid. Everything else is plumbing.

### Intermediate Representation

```python
@dataclass
class Cell:
    char: str
    fg: tuple[int, int, int] | None = None  # RGB
    bg: tuple[int, int, int] | None = None

@dataclass
class AsciiArt:
    width: int
    height: int
    cells: list[list[Cell]]
```

Every renderer consumes `AsciiArt`. Every mode (CLI, TUI, future web API) produces it through the same `core/engine.py` pipeline.

## Processing Pipeline

```
Load Image (Pillow: JPEG/PNG/WebP/BMP/GIF)
    │
    ▼
Preprocess (brightness, contrast, saturation, sharpness)
    │
    ▼
Resize + Aspect Correction (Lanczos downscale)
    height = width * (img_h / img_w) * font_ratio
    │
    ▼
Grayscale (BT.709 weighted luminance)
    (skipped in color mode — RGB preserved alongside)
    │
    ▼
Edge Detection (optional: Sobel → edge_mask + angles)
    │
    ▼
Dithering (optional: Floyd-Steinberg or ordered Bayer)
    │
    ▼
Character Mapping
    • Edge pixels → directional chars (| / \ -)
    • Normal pixels → index into character ramp by luminance
    • Braille mode → threshold 2x4 blocks into dot patterns
    • Block mode → map to ░▒▓█
    │
    ▼
Color Assignment (attach original RGB to each Cell)
    │
    ▼
AsciiArt Grid → any renderer
```

### Engine Signature

```python
def convert(image: Image.Image, options: ConvertOptions) -> AsciiArt:
```

## Rendering Modes

1. **Classic ASCII** — Pure printable ASCII characters (`@#%*+=-:. `), works everywhere
2. **Unicode blocks** — `█▓▒░` block characters for smooth density gradients
3. **Braille** — `⣿⡷⠛⠁` 2x4 dot grid per character, 4x the effective resolution
4. **Half-block color** — `▀▄` with fg/bg colors, 2x vertical resolution
5. **Colored** — Any mode above + truecolor ANSI (24-bit RGB per character)

## Interactive TUI (Textual)

Three-panel layout:

```
┌──────────────────────────────────────────────────────────┐
│  asciart - interactive mode                    [Q]uit    │
├────────────────────────────────┬─────────────────────────┤
│                                │  Mode  [ASCII ▾]        │
│                                │                         │
│                                │  Width ──●────── 100    │
│                                │  Font Ratio ──●── 0.5   │
│                                │                         │
│         LIVE PREVIEW           │  --- Image ---          │
│                                │  Brightness ──●── 0     │
│    (ASCII art renders here,    │  Contrast ────●── 1.0   │
│     updates on every           │  Saturation ──●── 1.0   │
│     slider change)             │  Sharpness ───●── 1.0   │
│                                │                         │
│                                │  --- Rendering ---      │
│                                │  Dither [None ▾]        │
│                                │  ☐ Edge Detection       │
│                                │  ☐ Invert               │
│                                │  Color  [Truecolor ▾]   │
│                                │                         │
│                                │  --- Export ---          │
│                                │  [Save TXT] [Save HTML] │
│                                │  [Save PNG] [Copy]      │
├────────────────────────────────┴─────────────────────────┤
│  80x40 chars | truecolor | 147ms render                  │
└──────────────────────────────────────────────────────────┘
```

- Debounced re-rendering: wait 100ms after last slider change before re-converting
- Preset buttons at top: [Photo] [Logo] [Retro] [HD] [Blocks] [Line Art]
- Status bar shows dimensions, color mode, render time
- GIF auto-play in preview with play/pause and frame scrubber

## CLI Mode (Typer)

Two subcommands:
- `asciart convert` — one-shot, flags-based, pipe-friendly
- `asciart play` — launches TUI

```bash
# Basic
asciart convert photo.jpg

# Full options
asciart convert photo.jpg -w 120 --color truecolor --mode braille --dither floyd-steinberg -o output.txt

# Animated GIF
asciart convert cat.gif --animate

# Interactive
asciart play photo.jpg

# Preset
asciart convert portrait.jpg --preset photo --color truecolor

# Export to HTML
asciart convert photo.jpg --format html -o art.html
```

## Export Formats

| Format | Flag | Output | Use Case |
|--------|------|--------|----------|
| text | --format text | Plain characters, no color | Paste anywhere |
| ansi | --format ansi | Characters + ANSI escapes | Terminal sharing |
| html | --format html | Self-contained HTML + CSS | Web sharing |
| png | --format png | Rendered image of ASCII art | Social media |
| svg | --format svg | Vector, resolution-independent | Docs, scaling |

Clipboard support via `--copy` (auto-detects pbcopy/xclip/clip).

## GIF Animation

### Terminal Playback
1. Extract all frames + durations from GIF
2. Batch pre-convert ALL frames to AsciiArt grids
3. Playback: cursor home (`\033[H`), print frame, sleep for delay, loop

### GIF Export
Render each frame to PIL Image, stitch into output GIF.

### TUI
Auto-play with play/pause button and frame scrubber.

## Preset System

```python
PRESETS = {
    "photo":   { mode: ASCII, chars: " .:-=+*#%@", color: TRUECOLOR, dither: FLOYD_STEINBERG, contrast: 1.2 },
    "logo":    { mode: BRAILLE, color: TRUECOLOR, contrast: 1.5, edge_detection: True },
    "retro":   { mode: ASCII, chars: " .:#", color: NONE, dither: ORDERED },
    "hd":      { mode: BRAILLE, color: TRUECOLOR, width: 200, dither: FLOYD_STEINBERG },
    "blocks":  { mode: HALFBLOCK, color: TRUECOLOR },
    "lineart": { mode: ASCII, edge_detection: True, color: NONE, contrast: 1.8 },
}
```

Custom presets saved to `~/.config/asciart/presets.toml`.

## Dependencies

| Package | Purpose |
|---------|---------|
| pillow>=10.0 | Image loading, resizing, preprocessing |
| numpy>=1.24 | Vectorized pixel math |
| typer>=0.9 | CLI framework |
| textual>=0.80 | TUI framework |
| rich>=13.0 | Terminal colors (Textual dependency) |

Dev: pytest, pytest-cov, ruff

## Setup (uv)

```bash
uv init asciart
uv add pillow numpy typer textual rich
uv add --dev pytest pytest-cov ruff
```

Entry point: `asciart = "asciart.cli:app"`

## Future Considerations

- **Web frontend**: Core engine has zero I/O deps, can be wrapped in a FastAPI endpoint or compiled to WASM via RustPython/Pyodide
- **GPU acceleration**: NumPy → CuPy swap for CUDA-accelerated pixel math
- **OpenCV integration**: Optional dep for CLAHE, advanced edge detection
- **Video/webcam**: OpenCV VideoCapture → frame-by-frame conversion
- **Custom fonts**: Structural similarity matching against user's terminal font

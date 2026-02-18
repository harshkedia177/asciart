# asciart

A high-performance image-to-ASCII art converter with GPU acceleration, perceptual color matching, and an interactive TUI.

```
                          @@@@@@@@@@@@@@
                     @@@@@..............@@@@@
                  @@@....::::::::::::::.....@@@
                @@..::::::=====+++====:::::::..@@
              @@.:::===+++***###***+++===:::.....@@
            @@.::==++**##%%@@@@@%%##**++==:::....@@
           @..::=++*##%%@@       @@%%##*++==::...@@
          @.::==+*##%@@             @@%##*+==::..@@
         @.::=++*#%@@                 @%##*+==:..@@
```

## Features

**4 rendering modes** to match any use case:

| Mode | Description | Best for |
|------|-------------|----------|
| `ascii` | Classic ASCII characters (`" .:-=+*#%@"`) | Terminal output, text files |
| `blocks` | Unicode block elements (`" ░▒▓█"`) | Dense, graphic-style output |
| `braille` | Unicode braille patterns (2x4 subpixel) | High-resolution detail |
| `halfblock` | Half-block characters with fg+bg color | Photo-realistic terminal art |

**3 character matching algorithms:**

| Mode | Approach | Speed | Quality |
|------|----------|-------|---------|
| `brightness` | Map pixel brightness to character ramp | Fastest | Good |
| `structural` | Match image tiles against pre-rendered glyph bitmaps (SSD) | Moderate | Better |
| `hybrid` | Multi-criteria: 25% brightness + 50% structure + 15% variance + 10% frequency | Slower | Best |

**5 dithering algorithms:** none, Floyd-Steinberg, ordered (8x8 Bayer), Atkinson, blue noise

**Color modes:** no color, ANSI-256 (perceptual CIELAB matching), truecolor (24-bit)

**Export formats:** ANSI terminal, plain text, HTML, SVG, PNG

**Acceleration:** auto-detects CuPy (GPU) > Numba (JIT) > NumPy, with 3-tier fallback

## Installation

Requires Python 3.11+.

```bash
# With uv (recommended)
uv pip install -e .

# With pip
pip install -e .
```

### Optional acceleration

```bash
# Numba JIT — 20-100x faster dithering
uv pip install -e ".[jit]"

# CuPy GPU — CUDA acceleration for large images
uv pip install -e ".[gpu]"

# Both
uv pip install -e ".[all]"
```

## Quick Start

```bash
# Basic conversion (80 chars wide, auto color)
asciart convert photo.jpg

# High-quality with structural matching and CLAHE
asciart convert photo.jpg --match hybrid --clahe --dither atkinson

# Braille mode for maximum detail
asciart convert photo.jpg --mode braille -w 200 --color truecolor

# Half-block mode for photo-realistic output
asciart convert photo.jpg --mode halfblock --color truecolor

# Save as HTML
asciart convert photo.jpg --format html -o output.html

# Use a preset
asciart convert photo.jpg --preset studio

# Animate a GIF in terminal
asciart convert animation.gif --animate

# Interactive TUI
asciart play photo.jpg

# TUI with file browser
asciart play
```

## CLI Reference

```
asciart convert [OPTIONS] IMAGE_PATH
```

### Core options

| Flag | Default | Description |
|------|---------|-------------|
| `-w, --width` | `80` | Output width in characters |
| `--mode` | `ascii` | Rendering mode: `ascii`, `blocks`, `braille`, `halfblock` |
| `--color` | `auto` | Color: `none`, `256`, `truecolor`, `auto` |
| `--match` | `brightness` | Matching: `brightness`, `structural`, `hybrid` |
| `--dither` | `none` | Dithering: `none`, `floyd-steinberg`, `ordered`, `atkinson`, `blue-noise` |
| `--format` | `ansi` | Output: `ansi`, `text`, `html`, `svg`, `png` |

### Image adjustments

| Flag | Default | Description |
|------|---------|-------------|
| `--brightness` | `0.0` | Brightness offset (-100 to 100) |
| `--contrast` | `1.0` | Contrast multiplier |
| `--ratio` | `0.5` | Font cell width/height ratio |
| `-i, --invert` | off | Invert brightness |
| `--clahe` | off | CLAHE local contrast enhancement |

### Output

| Flag | Description |
|------|-------------|
| `-o, --output` | Save to file (required for PNG format) |
| `--copy` | Copy to clipboard |
| `--animate` | Animate GIF frames in terminal |
| `--preset` | Use a preset (see below) |

### Advanced

| Flag | Description |
|------|-------------|
| `--font` | Custom font path for structural/hybrid matching |
| `--backend` | Force backend: `numpy`, `numba`, `cupy` |
| `-v, --verbose` | Show backend and timing info |

## Presets

Presets are curated combinations of settings for common use cases:

| Preset | Mode | Match | Color | Dither | Notes |
|--------|------|-------|-------|--------|-------|
| `photo` | ascii | brightness | truecolor | floyd-steinberg | General-purpose photos |
| `studio` | ascii | hybrid | truecolor | atkinson | Maximum quality |
| `logo` | braille | brightness | truecolor | none | Logos and line art with edge detection |
| `retro` | ascii | brightness | none | ordered | Monochrome retro aesthetic |
| `hd` | braille | brightness | truecolor | floyd-steinberg | High-resolution at 200 chars wide |
| `blocks` | halfblock | brightness | truecolor | none | Photo-realistic block art |
| `lineart` | ascii | brightness | none | none | Edge detection, high contrast |

```bash
asciart convert photo.jpg --preset studio
asciart convert logo.png --preset logo
```

## Interactive TUI

Launch an interactive terminal UI with live preview and all controls:

```bash
# Open with an image
asciart play photo.jpg

# Open file browser first
asciart play
```

The TUI provides:
- Live preview with debounced updates
- Sliders for width, brightness, contrast, saturation, sharpness, font ratio
- Dropdowns for mode, color, dither, match mode
- Toggles for CLAHE, edge detection, invert
- One-click presets
- Export to TXT, HTML, PNG
- Status bar showing dimensions, backend, and render time

## Architecture

```
Image → Preprocess → Resize → CLAHE → Grayscale → Dither → Match → AsciiArt → Render
         │              │        │                    │        │
         │              │        └─ adaptive local    │        ├─ brightness (fast)
         │              │           contrast          │        ├─ structural (SSD)
         │              │                             │        └─ hybrid (multi-criteria)
         │              └─ edge-preserving             │
         │                 (UnsharpMask + LANCZOS)     ├─ floyd-steinberg
         │                                             ├─ ordered (8x8 Bayer)
         └─ brightness, contrast,                      ├─ atkinson (75% diffusion)
            saturation, sharpness                      └─ blue noise
```

### Performance tiers

The engine auto-detects the best available backend:

| Tier | Backend | Speedup | Requirement |
|------|---------|---------|-------------|
| 1 | **NumPy** | Baseline | Always available |
| 2 | **Numba JIT** | 20-100x for dithering | `pip install numba` |
| 3 | **CuPy GPU** | 100x+ for large images | NVIDIA GPU + `pip install cupy-cuda12x` |

The core data model (`AsciiArt`) uses numpy arrays internally (`char_array`, `fg_array`, `bg_array`) rather than Python objects, enabling zero-copy handoff between pipeline stages.

### Color matching

ANSI-256 colors are matched using perceptual CIELAB Delta E distance rather than naive RGB Euclidean distance. The pipeline:

```
sRGB → Linear RGB → XYZ (D65) → L*a*b* → CIE76 Delta E → nearest ANSI-256
```

### Glyph matching

Structural and hybrid matching pre-render every ASCII character to a bitmap, extract 4x4 sub-block feature vectors, and match image tiles against glyphs using sum-of-squared-differences (SSD). Hybrid mode combines brightness, structural similarity, variance, and spatial frequency with learned weights.

## Project Structure

```
src/asciart/
├── cli.py                 # Typer CLI (convert, play)
├── tui.py                 # Textual interactive TUI
├── models.py              # AsciiArt, ConvertOptions, enums
├── presets.py             # Curated preset configurations
├── clipboard.py           # Clipboard integration
├── core/
│   ├── engine.py          # Main conversion pipeline
│   ├── preprocess.py      # Image preprocessing
│   ├── mapper.py          # Vectorized character mapping
│   ├── dither.py          # 5 dithering algorithms
│   ├── edges.py           # Sobel edge detection
│   ├── contrast.py        # CLAHE implementation
│   ├── color_space.py     # CIELAB perceptual color
│   ├── glyph_cache.py     # Structural glyph matching
│   ├── ramps.py           # Character ramp definitions
│   └── gif.py             # GIF frame extraction
├── renderers/
│   ├── terminal.py        # ANSI escape sequences
│   ├── text.py            # Plain text
│   ├── html.py            # Standalone HTML page
│   ├── svg.py             # Scalable vector graphics
│   └── image.py           # PNG via Pillow
└── accel/
    ├── backend.py         # Backend auto-detection
    └── numba_kernels.py   # JIT-compiled dithering
```

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/harshkedia177/asciart.git
cd asciart
uv sync --group dev

# Run tests (160+ tests)
uv run pytest

# Run with coverage
uv run pytest --cov=asciart

# Lint
uv run ruff check src/ tests/
```

## Requirements

- Python >= 3.11
- pillow >= 10.0
- numpy >= 1.24
- typer >= 0.9
- textual >= 0.80
- rich >= 13.0
- scipy >= 1.11
- textual-slider >= 0.2.0

## License

MIT

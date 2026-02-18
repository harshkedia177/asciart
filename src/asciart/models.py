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

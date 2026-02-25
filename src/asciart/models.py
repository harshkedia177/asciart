from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


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
    ATKINSON = "atkinson"
    BLUE_NOISE = "blue-noise"


class MatchMode(str, Enum):
    BRIGHTNESS = "brightness"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"


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


class AsciiArt:
    """The intermediate representation -- output of core engine, input to renderers.

    Supports two construction modes:
      1. Legacy: AsciiArt(width, height, cells=[[Cell(...)]])
      2. Array-based: AsciiArt.from_arrays(char_array, char_map, fg_array, bg_array)

    In both cases, the `cells` property returns list[list[Cell]] for backward compat.
    """

    __slots__ = (
        "width",
        "height",
        "_cells",
        "char_array",
        "char_map",
        "fg_array",
        "bg_array",
        "_array_mode",
    )

    def __init__(
        self,
        width: int,
        height: int,
        cells: list[list[Cell]] | None = None,
        *,
        char_array: np.ndarray | None = None,
        char_map: str | None = None,
        fg_array: np.ndarray | None = None,
        bg_array: np.ndarray | None = None,
    ) -> None:
        self.width = width
        self.height = height

        if char_array is not None:
            # Array-based construction
            self._array_mode = True
            self.char_array = char_array
            self.char_map = char_map if char_map is not None else ""
            self.fg_array = fg_array
            self.bg_array = bg_array
            self._cells = None
        else:
            # Legacy cell-based construction
            self._array_mode = False
            self._cells = cells if cells is not None else []
            self.char_array = None
            self.char_map = None
            self.fg_array = None
            self.bg_array = None

    @classmethod
    def from_arrays(
        cls,
        char_array: np.ndarray,
        char_map: str,
        fg_array: np.ndarray | None = None,
        bg_array: np.ndarray | None = None,
    ) -> AsciiArt:
        """Create an AsciiArt from numpy arrays.

        Args:
            char_array: H x W uint8 array of indices into char_map.
            char_map: The character ramp string (e.g. " .:-=+*#%@").
            fg_array: Optional H x W x 3 uint8 array of foreground RGB colors.
            bg_array: Optional H x W x 3 uint8 array of background RGB colors.
        """
        h, w = char_array.shape[:2]
        return cls(
            width=w,
            height=h,
            char_array=char_array,
            char_map=char_map,
            fg_array=fg_array,
            bg_array=bg_array,
        )

    @property
    def cells(self) -> list[list[Cell]]:
        """Backward-compatible property that returns list[list[Cell]].

        If constructed from arrays, reconstructs Cell objects on access.
        If constructed from legacy cells, returns them directly.
        """
        if not self._array_mode:
            return self._cells  # type: ignore[return-value]

        # Reconstruct from arrays
        rows: list[list[Cell]] = []
        char_map = self.char_map or ""
        char_array = self.char_array
        fg_array = self.fg_array
        bg_array = self.bg_array

        for y in range(self.height):
            row: list[Cell] = []
            for x in range(self.width):
                idx = int(char_array[y, x])  # type: ignore[index]
                char = char_map[idx] if idx < len(char_map) else " "

                fg = None
                if fg_array is not None:
                    r, g, b = fg_array[y, x]  # type: ignore[index]
                    fg = (int(r), int(g), int(b))

                bg = None
                if bg_array is not None:
                    r, g, b = bg_array[y, x]  # type: ignore[index]
                    bg = (int(r), int(g), int(b))

                row.append(Cell(char=char, fg=fg, bg=bg))
            rows.append(row)
        return rows

    @cells.setter
    def cells(self, value: list[list[Cell]]) -> None:
        """Allow setting cells directly (legacy compatibility)."""
        self._cells = value
        self._array_mode = False

    def __repr__(self) -> str:
        return f"AsciiArt(width={self.width}, height={self.height})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AsciiArt):
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.cells == other.cells
        )


@dataclass
class ConvertOptions:
    """All settings for a conversion. Every slider/flag maps to a field here."""
    width: int = 80
    height: int | None = None
    mode: Mode = Mode.ASCII
    chars: str = "standard"
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
    match_mode: MatchMode = MatchMode.BRIGHTNESS
    clahe: bool = False
    font_path: str | None = None

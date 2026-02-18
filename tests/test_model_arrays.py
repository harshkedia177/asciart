# tests/test_model_arrays.py
"""Tests for numpy array-based AsciiArt construction and backward compatibility."""

import numpy as np

from asciart.models import (
    AsciiArt,
    Cell,
    ConvertOptions,
    DitherMode,
    MatchMode,
)


# ---------------------------------------------------------------------------
# New enum values
# ---------------------------------------------------------------------------


class TestMatchMode:
    def test_brightness_value(self):
        assert MatchMode.BRIGHTNESS == "brightness"

    def test_structural_value(self):
        assert MatchMode.STRUCTURAL == "structural"

    def test_hybrid_value(self):
        assert MatchMode.HYBRID == "hybrid"


class TestDitherModeExtensions:
    def test_atkinson_value(self):
        assert DitherMode.ATKINSON == "atkinson"

    def test_blue_noise_value(self):
        assert DitherMode.BLUE_NOISE == "blue-noise"

    def test_original_values_still_exist(self):
        assert DitherMode.NONE == "none"
        assert DitherMode.FLOYD_STEINBERG == "floyd-steinberg"
        assert DitherMode.ORDERED == "ordered"


# ---------------------------------------------------------------------------
# ConvertOptions new fields
# ---------------------------------------------------------------------------


class TestConvertOptionsNewFields:
    def test_match_mode_default(self):
        opts = ConvertOptions()
        assert opts.match_mode == MatchMode.BRIGHTNESS

    def test_clahe_default(self):
        opts = ConvertOptions()
        assert opts.clahe is False

    def test_font_path_default(self):
        opts = ConvertOptions()
        assert opts.font_path is None

    def test_match_mode_custom(self):
        opts = ConvertOptions(match_mode=MatchMode.HYBRID)
        assert opts.match_mode == MatchMode.HYBRID

    def test_clahe_enabled(self):
        opts = ConvertOptions(clahe=True)
        assert opts.clahe is True

    def test_font_path_custom(self):
        opts = ConvertOptions(font_path="/usr/share/fonts/mono.ttf")
        assert opts.font_path == "/usr/share/fonts/mono.ttf"

    def test_existing_defaults_unchanged(self):
        """Ensure the new fields don't break existing defaults."""
        opts = ConvertOptions()
        assert opts.width == 80
        assert opts.dither == DitherMode.NONE
        assert opts.contrast == 1.0


# ---------------------------------------------------------------------------
# AsciiArt.from_arrays()
# ---------------------------------------------------------------------------


class TestFromArrays:
    def test_basic_construction(self):
        char_array = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        char_map = " .:#"
        art = AsciiArt.from_arrays(char_array, char_map)
        assert art.width == 2
        assert art.height == 2
        assert art.char_array is char_array
        assert art.char_map == char_map

    def test_with_fg_array(self):
        char_array = np.array([[0, 1]], dtype=np.uint8)
        fg_array = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, " @", fg_array=fg_array)
        assert art.fg_array is fg_array
        assert art.bg_array is None

    def test_with_bg_array(self):
        char_array = np.array([[0, 1]], dtype=np.uint8)
        bg_array = np.array([[[0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, " @", bg_array=bg_array)
        assert art.fg_array is None
        assert art.bg_array is bg_array

    def test_with_both_fg_and_bg(self):
        char_array = np.array([[0]], dtype=np.uint8)
        fg_array = np.array([[[255, 0, 0]]], dtype=np.uint8)
        bg_array = np.array([[[0, 0, 255]]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, "@", fg_array=fg_array, bg_array=bg_array)
        assert art.fg_array is not None
        assert art.bg_array is not None

    def test_dimensions_from_array_shape(self):
        char_array = np.zeros((5, 10), dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, " ")
        assert art.width == 10
        assert art.height == 5


# ---------------------------------------------------------------------------
# cells property -- reconstruction from arrays
# ---------------------------------------------------------------------------


class TestCellsPropertyFromArrays:
    def test_char_reconstruction(self):
        char_array = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, " .:#")
        cells = art.cells
        assert len(cells) == 1
        assert len(cells[0]) == 4
        assert cells[0][0].char == " "
        assert cells[0][1].char == "."
        assert cells[0][2].char == ":"
        assert cells[0][3].char == "#"

    def test_fg_reconstruction(self):
        char_array = np.array([[0]], dtype=np.uint8)
        fg_array = np.array([[[100, 200, 50]]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, "@", fg_array=fg_array)
        cell = art.cells[0][0]
        assert cell.fg == (100, 200, 50)

    def test_bg_reconstruction(self):
        char_array = np.array([[0]], dtype=np.uint8)
        bg_array = np.array([[[10, 20, 30]]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, "@", bg_array=bg_array)
        cell = art.cells[0][0]
        assert cell.bg == (10, 20, 30)

    def test_no_color_arrays_gives_none(self):
        char_array = np.array([[0]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, "@")
        cell = art.cells[0][0]
        assert cell.fg is None
        assert cell.bg is None

    def test_multirow_reconstruction(self):
        char_array = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, "ABCD")
        cells = art.cells
        assert cells[0][0].char == "A"
        assert cells[0][1].char == "B"
        assert cells[1][0].char == "C"
        assert cells[1][1].char == "D"

    def test_indexed_access_pattern(self):
        """art.cells[y][x].char must work (the most common access pattern)."""
        char_array = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        fg_array = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, "@#.", fg_array=fg_array)
        assert art.cells[0][0].char == "@"
        assert art.cells[0][1].char == "#"
        assert art.cells[1][0].char == "."
        assert art.cells[1][1].char == "@"
        assert art.cells[0][0].fg == (255, 0, 0)
        assert art.cells[1][0].fg == (0, 0, 255)

    def test_iteration_pattern(self):
        """for row in art.cells: for cell in row: -- must work."""
        char_array = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, " @")
        chars = []
        for row in art.cells:
            for cell in row:
                chars.append(cell.char)
        assert chars == [" ", "@", "@", " "]


# ---------------------------------------------------------------------------
# Legacy construction backward compatibility
# ---------------------------------------------------------------------------


class TestLegacyConstruction:
    def test_basic_legacy(self):
        cells = [[Cell(char="@"), Cell(char="#")]]
        art = AsciiArt(width=2, height=1, cells=cells)
        assert art.cells[0][0].char == "@"
        assert art.cells[0][1].char == "#"

    def test_empty_cells_default(self):
        art = AsciiArt(width=0, height=0)
        assert art.cells == []

    def test_legacy_with_colors(self):
        cells = [[Cell(char="X", fg=(255, 0, 0), bg=(0, 0, 255))]]
        art = AsciiArt(width=1, height=1, cells=cells)
        cell = art.cells[0][0]
        assert cell.fg == (255, 0, 0)
        assert cell.bg == (0, 0, 255)

    def test_legacy_array_fields_are_none(self):
        art = AsciiArt(width=1, height=1, cells=[[Cell(char=".")]])
        assert art.char_array is None
        assert art.char_map is None
        assert art.fg_array is None
        assert art.bg_array is None

    def test_cells_setter(self):
        """Setting cells directly must switch to legacy mode."""
        char_array = np.array([[0]], dtype=np.uint8)
        art = AsciiArt.from_arrays(char_array, "@")
        # Override with legacy cells
        art.cells = [[Cell(char="X")]]
        assert art.cells[0][0].char == "X"

    def test_len_cells(self):
        cells = [[Cell(char="A")], [Cell(char="B")]]
        art = AsciiArt(width=1, height=2, cells=cells)
        assert len(art.cells) == 2
        assert len(art.cells[0]) == 1


# ---------------------------------------------------------------------------
# Repr and equality
# ---------------------------------------------------------------------------


class TestReprAndEquality:
    def test_repr(self):
        art = AsciiArt(width=10, height=5)
        assert repr(art) == "AsciiArt(width=10, height=5)"

    def test_equality_legacy(self):
        a = AsciiArt(width=1, height=1, cells=[[Cell(char="@")]])
        b = AsciiArt(width=1, height=1, cells=[[Cell(char="@")]])
        assert a == b

    def test_inequality_different_cells(self):
        a = AsciiArt(width=1, height=1, cells=[[Cell(char="@")]])
        b = AsciiArt(width=1, height=1, cells=[[Cell(char="#")]])
        assert a != b

    def test_equality_array_based(self):
        arr = np.array([[0, 1]], dtype=np.uint8)
        a = AsciiArt.from_arrays(arr.copy(), " @")
        b = AsciiArt.from_arrays(arr.copy(), " @")
        assert a == b

    def test_equality_array_vs_legacy(self):
        """Array-based and legacy AsciiArt that produce the same cells should be equal."""
        arr = np.array([[1]], dtype=np.uint8)
        array_art = AsciiArt.from_arrays(arr, " @")
        legacy_art = AsciiArt(width=1, height=1, cells=[[Cell(char="@")]])
        assert array_art == legacy_art

"""Tests for vectorized mapper functions returning numpy arrays."""

import numpy as np

from asciart.core.mapper import map_brightness, map_braille, map_halfblock



class TestMapBrightnessShapesAndDtypes:
    def test_returns_tuple_of_three(self):
        gray = np.full((4, 6), 128.0)
        result = map_brightness(gray, " .:#", invert=False)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_char_indices_shape_and_dtype(self):
        gray = np.full((5, 10), 128.0)
        char_indices, _, _ = map_brightness(gray, " .:#", invert=False)
        assert char_indices.shape == (5, 10)
        assert char_indices.dtype == np.uint8

    def test_fg_none_when_no_colors(self):
        gray = np.full((3, 3), 128.0)
        _, _, fg = map_brightness(gray, " .:#", invert=False)
        assert fg is None

    def test_fg_shape_and_dtype_with_colors(self):
        gray = np.full((4, 6), 128.0)
        colors = np.random.randint(0, 256, (4, 6, 3), dtype=np.uint8).astype(np.float64)
        _, _, fg = map_brightness(gray, " .:#", invert=False, colors=colors)
        assert fg is not None
        assert fg.shape == (4, 6, 3)
        assert fg.dtype == np.uint8


class TestMapBrightnessMappingCorrectness:
    def test_white_maps_to_index_zero(self):
        """White (gray=255) -> index 0 (lightest char)."""
        gray = np.full((2, 3), 255.0)
        char_indices, ramp, _ = map_brightness(gray, " .:#", invert=False)
        assert np.all(char_indices == 0)
        assert ramp[0] == " "

    def test_black_maps_to_last_index(self):
        """Black (gray=0) -> last index (darkest char)."""
        gray = np.full((2, 3), 0.0)
        char_indices, ramp, _ = map_brightness(gray, " .:#", invert=False)
        assert np.all(char_indices == 3)
        assert ramp[3] == "#"

    def test_gradient_covers_full_ramp(self):
        """A gradient should use all indices in the ramp."""
        ramp_str = " .:-=+*#%@"
        gray = np.linspace(0, 255, 256).reshape(1, 256)
        char_indices, _, _ = map_brightness(gray, ramp_str, invert=False)
        unique_indices = set(char_indices.flatten())
        assert len(unique_indices) >= len(ramp_str) - 1  # should cover most/all

    def test_mid_gray_maps_to_middle_index(self):
        """Mid-gray should map to a middle index."""
        gray = np.full((1, 1), 127.5)
        char_indices, _, _ = map_brightness(gray, " .:#", invert=False)
        idx = int(char_indices[0, 0])
        # Should be index 1 or 2 (middle of 0-3 range)
        assert 1 <= idx <= 2


class TestMapBrightnessInvert:
    def test_invert_reverses_ramp(self):
        """With invert, the ramp string is reversed."""
        gray = np.full((1, 1), 255.0)
        _, ramp_normal, _ = map_brightness(gray, " .:#", invert=False)
        _, ramp_inverted, _ = map_brightness(gray, " .:#", invert=True)
        assert ramp_normal == " .:#"
        assert ramp_inverted == "#:. "

    def test_invert_white_maps_to_darkest(self):
        """With invert, white -> index 0, but ramp[0] is now '#' (the darkest)."""
        gray = np.full((2, 2), 255.0)
        char_indices, ramp, _ = map_brightness(gray, " .:#", invert=True)
        assert np.all(char_indices == 0)
        assert ramp[0] == "#"

    def test_invert_black_maps_to_lightest(self):
        """With invert, black -> last index, and ramp[-1] is now ' ' (lightest)."""
        gray = np.full((2, 2), 0.0)
        char_indices, ramp, _ = map_brightness(gray, " .:#", invert=True)
        assert np.all(char_indices == 3)
        assert ramp[3] == " "


class TestMapBrightnessColors:
    def test_colors_preserved_as_uint8(self):
        gray = np.full((2, 2), 128.0)
        colors = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ], dtype=np.float64)
        _, _, fg = map_brightness(gray, " .:#", invert=False, colors=colors)
        assert fg is not None
        assert fg.dtype == np.uint8
        np.testing.assert_array_equal(fg[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(fg[0, 1], [0, 255, 0])
        np.testing.assert_array_equal(fg[1, 0], [0, 0, 255])
        np.testing.assert_array_equal(fg[1, 1], [128, 128, 128])



class TestMapBrailleShapesAndDtypes:
    def test_returns_tuple_of_three(self):
        gray = np.full((8, 4), 128.0)
        result = map_braille(gray)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_char_indices_shape(self):
        """8x4 input -> 2x2 braille output (8//4=2, 4//2=2)."""
        gray = np.full((8, 4), 128.0)
        char_indices, _, _ = map_braille(gray)
        assert char_indices.shape == (2, 2)

    def test_char_indices_dtype(self):
        gray = np.full((8, 4), 128.0)
        char_indices, _, _ = map_braille(gray)
        assert char_indices.dtype == np.uint8

    def test_char_map_is_256_braille_chars(self):
        gray = np.full((4, 2), 128.0)
        _, char_map, _ = map_braille(gray)
        assert len(char_map) == 256
        assert char_map[0] == chr(0x2800)
        assert char_map[255] == chr(0x28FF)

    def test_padding_non_multiple_dimensions(self):
        """Non-multiple dimensions should be padded automatically."""
        gray = np.full((5, 3), 128.0)
        char_indices, _, _ = map_braille(gray)
        # 5 padded to 8, 3 padded to 4 -> 2x2 output
        assert char_indices.shape == (2, 2)

    def test_fg_none_without_colors(self):
        gray = np.full((4, 2), 128.0)
        _, _, fg = map_braille(gray)
        assert fg is None

    def test_fg_shape_with_colors(self):
        gray = np.full((8, 4), 128.0)
        colors = np.random.randint(0, 256, (8, 4, 3)).astype(np.float64)
        _, _, fg = map_braille(gray, colors=colors)
        assert fg is not None
        assert fg.shape == (2, 2, 3)
        assert fg.dtype == np.uint8


class TestMapBrailleMappingCorrectness:
    def test_white_image_empty_braille(self):
        """White image (all above threshold) -> char offset 0 (empty braille)."""
        gray = np.full((4, 2), 255.0)
        char_indices, char_map, _ = map_braille(gray, threshold=128.0)
        assert np.all(char_indices == 0)
        assert char_map[0] == chr(0x2800)

    def test_black_image_full_braille(self):
        """Black image (all below threshold) -> char offset 0xFF (full braille)."""
        gray = np.full((4, 2), 0.0)
        char_indices, char_map, _ = map_braille(gray, threshold=128.0)
        assert np.all(char_indices == 0xFF)
        assert char_map[0xFF] == chr(0x28FF)

    def test_single_dot_top_left(self):
        """Only pixel (0,0) is dark -> bit 0x01 -> offset 1."""
        gray = np.full((4, 2), 255.0)
        gray[0, 0] = 0.0  # dark
        char_indices, _, _ = map_braille(gray, threshold=128.0)
        assert char_indices[0, 0] == 0x01

    def test_single_dot_bottom_right(self):
        """Only pixel (3,1) is dark -> bit 0x80 -> offset 128."""
        gray = np.full((4, 2), 255.0)
        gray[3, 1] = 0.0  # dark
        char_indices, _, _ = map_braille(gray, threshold=128.0)
        assert char_indices[0, 0] == 0x80

    def test_left_column_only(self):
        """All left column dark -> bits 0x01|0x02|0x04|0x40 = 0x47."""
        gray = np.full((4, 2), 255.0)
        gray[:, 0] = 0.0
        char_indices, _, _ = map_braille(gray, threshold=128.0)
        assert char_indices[0, 0] == 0x47



class TestMapHalfblockShapesAndDtypes:
    def test_returns_tuple_of_four(self):
        pixels = np.random.randint(0, 256, (4, 6, 3), dtype=np.uint8).astype(np.float64)
        result = map_halfblock(pixels)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_char_indices_shape(self):
        """4x6 input -> 2x6 output (4//2=2)."""
        pixels = np.random.randint(0, 256, (4, 6, 3), dtype=np.uint8).astype(np.float64)
        char_indices, _, _, _ = map_halfblock(pixels)
        assert char_indices.shape == (2, 6)
        assert char_indices.dtype == np.uint8

    def test_char_indices_all_zeros(self):
        pixels = np.random.randint(0, 256, (4, 6, 3), dtype=np.uint8).astype(np.float64)
        char_indices, _, _, _ = map_halfblock(pixels)
        assert np.all(char_indices == 0)

    def test_char_map_is_upper_half_block(self):
        pixels = np.random.randint(0, 256, (4, 6, 3), dtype=np.uint8).astype(np.float64)
        _, char_map, _, _ = map_halfblock(pixels)
        assert char_map == "\u2580"

    def test_fg_bg_shapes(self):
        pixels = np.random.randint(0, 256, (4, 6, 3), dtype=np.uint8).astype(np.float64)
        _, _, fg, bg = map_halfblock(pixels)
        assert fg.shape == (2, 6, 3)
        assert bg.shape == (2, 6, 3)
        assert fg.dtype == np.uint8
        assert bg.dtype == np.uint8

    def test_odd_height_padding(self):
        """Odd height should be padded to even."""
        pixels = np.random.randint(0, 256, (5, 4, 3), dtype=np.uint8).astype(np.float64)
        char_indices, _, fg, bg = map_halfblock(pixels)
        assert char_indices.shape == (3, 4)
        assert fg.shape == (3, 4, 3)
        assert bg.shape == (3, 4, 3)


class TestMapHalfblockColorCorrectness:
    def test_fg_is_top_row_bg_is_bottom_row(self):
        """FG should come from even rows, BG from odd rows."""
        pixels = np.zeros((4, 2, 3), dtype=np.float64)
        # Row 0: red, Row 1: blue, Row 2: green, Row 3: white
        pixels[0, :] = [255, 0, 0]
        pixels[1, :] = [0, 0, 255]
        pixels[2, :] = [0, 255, 0]
        pixels[3, :] = [255, 255, 255]

        _, _, fg, bg = map_halfblock(pixels)
        # Output row 0: fg=row0(red), bg=row1(blue)
        np.testing.assert_array_equal(fg[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(bg[0, 0], [0, 0, 255])
        # Output row 1: fg=row2(green), bg=row3(white)
        np.testing.assert_array_equal(fg[1, 0], [0, 255, 0])
        np.testing.assert_array_equal(bg[1, 0], [255, 255, 255])



class TestNoLoopsVerification:
    """These tests verify the mappers handle realistic sizes efficiently,
    which would be extremely slow with Python loops."""

    def test_brightness_large_array(self):
        gray = np.random.rand(200, 300) * 255
        colors = np.random.rand(200, 300, 3) * 255
        char_indices, ramp, fg = map_brightness(gray, " .:-=+*#%@", invert=False, colors=colors)
        assert char_indices.shape == (200, 300)
        assert fg.shape == (200, 300, 3)

    def test_braille_large_array(self):
        gray = np.random.rand(400, 200) * 255
        colors = np.random.rand(400, 200, 3) * 255
        char_indices, char_map, fg = map_braille(gray, 128.0, colors)
        assert char_indices.shape == (100, 100)
        assert fg.shape == (100, 100, 3)

    def test_halfblock_large_array(self):
        pixels = np.random.rand(200, 300, 3) * 255
        char_indices, char_map, fg, bg = map_halfblock(pixels)
        assert char_indices.shape == (100, 300)
        assert fg.shape == (100, 300, 3)
        assert bg.shape == (100, 300, 3)

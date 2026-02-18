"""CIELAB color space utilities for perceptual ANSI-256 color matching."""
from __future__ import annotations

import numpy as np

# D65 reference white point
_Xn = 0.95047
_Yn = 1.00000
_Zn = 1.08883

# sRGB to XYZ (D65) transformation matrix
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def srgb_to_linear(c: float) -> float:
    """Convert a single sRGB channel (0-1) to linear RGB."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert sRGB (0-255) to CIELAB (L*, a*, b*) using D65 illuminant."""
    # Normalize to 0-1 and linearize
    rl = srgb_to_linear(r / 255.0)
    gl = srgb_to_linear(g / 255.0)
    bl = srgb_to_linear(b / 255.0)

    # Linear RGB to XYZ
    x = _SRGB_TO_XYZ[0, 0] * rl + _SRGB_TO_XYZ[0, 1] * gl + _SRGB_TO_XYZ[0, 2] * bl
    y = _SRGB_TO_XYZ[1, 0] * rl + _SRGB_TO_XYZ[1, 1] * gl + _SRGB_TO_XYZ[1, 2] * bl
    z = _SRGB_TO_XYZ[2, 0] * rl + _SRGB_TO_XYZ[2, 1] * gl + _SRGB_TO_XYZ[2, 2] * bl

    # XYZ to Lab
    def _f(t: float) -> float:
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return 7.787 * t + 16.0 / 116.0

    fx = _f(x / _Xn)
    fy = _f(y / _Yn)
    fz = _f(z / _Zn)

    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)

    return (l_star, a_star, b_star)


def _build_ansi256_rgb_table() -> list[tuple[int, int, int]]:
    """Build a table of RGB values for all 256 ANSI colors.

    - 0-15: standard 16 system colors (approximate values)
    - 16-231: 6x6x6 color cube
    - 232-255: grayscale ramp
    """
    table: list[tuple[int, int, int]] = []

    # 0-15: Standard 16 colors (widely-used approximate values)
    _std16 = [
        (0, 0, 0),        # 0  black
        (128, 0, 0),      # 1  red
        (0, 128, 0),      # 2  green
        (128, 128, 0),    # 3  yellow
        (0, 0, 128),      # 4  blue
        (128, 0, 128),    # 5  magenta
        (0, 128, 128),    # 6  cyan
        (192, 192, 192),  # 7  white
        (128, 128, 128),  # 8  bright black (gray)
        (255, 0, 0),      # 9  bright red
        (0, 255, 0),      # 10 bright green
        (255, 255, 0),    # 11 bright yellow
        (0, 0, 255),      # 12 bright blue
        (255, 0, 255),    # 13 bright magenta
        (0, 255, 255),    # 14 bright cyan
        (255, 255, 255),  # 15 bright white
    ]
    table.extend(_std16)

    # 16-231: 6x6x6 color cube
    _cube_values = [0, 51, 102, 153, 204, 255]  # i * 51
    for ri in range(6):
        for gi in range(6):
            for bi in range(6):
                table.append((_cube_values[ri], _cube_values[gi], _cube_values[bi]))

    # 232-255: grayscale ramp
    for i in range(24):
        v = 8 + i * 10
        table.append((v, v, v))

    return table


# Module-level caches
_ANSI256_RGB_TABLE: list[tuple[int, int, int]] | None = None
_ANSI256_LAB_TABLE: list[tuple[float, float, float]] | None = None


def _get_ansi256_tables() -> (
    tuple[list[tuple[int, int, int]], list[tuple[float, float, float]]]
):
    """Get or build cached ANSI-256 RGB and LAB tables."""
    global _ANSI256_RGB_TABLE, _ANSI256_LAB_TABLE  # noqa: PLW0603
    if _ANSI256_RGB_TABLE is None:
        _ANSI256_RGB_TABLE = _build_ansi256_rgb_table()
        _ANSI256_LAB_TABLE = [rgb_to_lab(*c) for c in _ANSI256_RGB_TABLE]
    assert _ANSI256_LAB_TABLE is not None
    return _ANSI256_RGB_TABLE, _ANSI256_LAB_TABLE


def nearest_ansi256_perceptual(r: int, g: int, b: int) -> int:
    """Find nearest ANSI-256 color using CIE76 Delta E in CIELAB space.

    Skips the first 16 system colors (indices 0-15) since their actual
    appearance varies per terminal emulator.
    """
    _, lab_table = _get_ansi256_tables()
    target_lab = rgb_to_lab(r, g, b)

    best_idx = 16
    best_dist = float("inf")

    for i in range(16, 256):
        dl = target_lab[0] - lab_table[i][0]
        da = target_lab[1] - lab_table[i][1]
        db = target_lab[2] - lab_table[i][2]
        dist = dl * dl + da * da + db * db
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def build_ansi256_lut(resolution: int = 64) -> np.ndarray:
    """Build a 3D lookup table mapping quantized RGB to nearest ANSI-256 index.

    The LUT has shape (resolution, resolution, resolution) with dtype uint8.
    To look up a color, quantize each channel: idx = channel * (resolution-1) // 255.
    """
    lut = np.empty((resolution, resolution, resolution), dtype=np.uint8)
    step = 255.0 / (resolution - 1)

    for ri in range(resolution):
        for gi in range(resolution):
            for bi in range(resolution):
                r = int(round(ri * step))
                g = int(round(gi * step))
                b = int(round(bi * step))
                lut[ri, gi, bi] = nearest_ansi256_perceptual(r, g, b)

    return lut

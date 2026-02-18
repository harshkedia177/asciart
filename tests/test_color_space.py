# tests/test_color_space.py
"""Tests for CIELAB color space utilities and perceptual ANSI-256 matching."""
from __future__ import annotations

import numpy as np

from asciart.core.color_space import (
    build_ansi256_lut,
    nearest_ansi256_perceptual,
    rgb_to_lab,
)


def test_rgb_to_lab_black():
    """Black should have L* near 0."""
    lstar, a, b = rgb_to_lab(0, 0, 0)
    assert abs(lstar) < 1.0, f"Expected L* near 0 for black, got {lstar}"


def test_rgb_to_lab_white():
    """White should have L* near 100."""
    lstar, a, b = rgb_to_lab(255, 255, 255)
    assert abs(lstar - 100.0) < 1.0, f"Expected L* near 100 for white, got {lstar}"


def test_rgb_to_lab_red_has_positive_a():
    """Pure red should have positive a* (red-green axis)."""
    _lstar, a, b = rgb_to_lab(255, 0, 0)
    assert a > 0, f"Expected positive a* for red, got {a}"


def test_nearest_ansi256_black():
    """Black should map to a valid index in the 16-255 range."""
    idx = nearest_ansi256_perceptual(0, 0, 0)
    assert 16 <= idx <= 255, f"Expected index in 16-255, got {idx}"


def test_nearest_ansi256_white():
    """White should map to index 231 (cube white) or a high grayscale index."""
    idx = nearest_ansi256_perceptual(255, 255, 255)
    # 231 is the 6x6x6 cube white (255,255,255), also acceptable are high grayscales
    assert idx == 231 or idx >= 248, f"Expected 231 or high grayscale for white, got {idx}"


def test_nearest_ansi256_gray():
    """A mid-gray should prefer the grayscale ramp (indices 232-255)."""
    idx = nearest_ansi256_perceptual(128, 128, 128)
    assert idx >= 232, f"Expected grayscale ramp index (>= 232) for gray, got {idx}"


def test_build_lut_shape():
    """LUT should have the correct shape and dtype."""
    res = 8  # small resolution for speed
    lut = build_ansi256_lut(resolution=res)
    assert lut.shape == (res, res, res), f"Expected shape ({res},{res},{res}), got {lut.shape}"
    assert lut.dtype == np.uint8, f"Expected dtype uint8, got {lut.dtype}"


def test_cielab_roundtrip_sanity():
    """A few known colors should produce reasonable L* values."""
    # Black: L* near 0
    l_black, _, _ = rgb_to_lab(0, 0, 0)
    assert l_black < 5.0

    # White: L* near 100
    l_white, _, _ = rgb_to_lab(255, 255, 255)
    assert l_white > 95.0

    # Mid-gray: L* roughly around 50-55
    l_gray, _, _ = rgb_to_lab(119, 119, 119)
    assert 40.0 < l_gray < 60.0, f"Expected L* ~50 for mid-gray, got {l_gray}"

    # Pure green: moderate L* (brighter than red or blue)
    l_green, _, _ = rgb_to_lab(0, 255, 0)
    assert 80.0 < l_green < 95.0, f"Expected L* 80-95 for green, got {l_green}"

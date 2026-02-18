import numpy as np
from asciart.core.dither import (
    atkinson,
    blue_noise_dither,
    ordered_dither,
    BAYER_8X8,
)


def test_atkinson_returns_same_shape():
    img = np.random.rand(10, 20) * 255
    result = atkinson(img, num_levels=4)
    assert result.shape == (10, 20)


def test_atkinson_output_is_quantized():
    img = np.random.rand(10, 20) * 255
    num_levels = 4
    result = atkinson(img, num_levels=num_levels)
    expected_values = {round(i / (num_levels - 1) * 255.0) for i in range(num_levels)}
    unique = set(np.round(result).astype(int).ravel())
    assert unique.issubset(expected_values), (
        f"Output contains unexpected values: {unique - expected_values}"
    )


def test_atkinson_preserves_highlights():
    """White input should stay mostly white (mean > 200)."""
    img = np.full((20, 20), 255.0)
    result = atkinson(img, num_levels=4)
    assert result.mean() > 200, f"Mean was {result.mean()}, expected > 200"


def test_blue_noise_returns_same_shape():
    img = np.random.rand(10, 20) * 255
    result = blue_noise_dither(img, num_levels=4)
    assert result.shape == (10, 20)


def test_blue_noise_output_valid_range():
    img = np.random.rand(10, 20) * 255
    result = blue_noise_dither(img, num_levels=4)
    assert result.min() >= 0, f"Min value {result.min()} is below 0"
    assert result.max() <= 255, f"Max value {result.max()} exceeds 255"


def test_blue_noise_output_is_quantized():
    img = np.random.rand(10, 20) * 255
    num_levels = 4
    result = blue_noise_dither(img, num_levels=num_levels)
    expected_values = {round(i / (num_levels - 1) * 255.0) for i in range(num_levels)}
    unique = set(np.round(result).astype(int).ravel())
    assert unique.issubset(expected_values), (
        f"Output contains unexpected values: {unique - expected_values}"
    )


def test_bayer_8x8_exists_and_correct_shape():
    assert BAYER_8X8.shape == (8, 8)


def test_ordered_dither_still_works():
    img = np.random.rand(16, 16) * 255
    result = ordered_dither(img, num_levels=4)
    assert result.shape == (16, 16)
    unique = np.unique(np.round(result).astype(int))
    assert len(unique) <= 4

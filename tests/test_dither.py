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
    assert len(unique) <= 4

def test_floyd_steinberg_uniform_stays_uniform():
    img = np.full((10, 10), 128.0)
    result = floyd_steinberg(img, num_levels=10)
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
    gradient = np.tile(np.linspace(0, 255, 100), (10, 1))
    dithered = floyd_steinberg(gradient, num_levels=4)
    naive = np.round(gradient / 255 * 3) / 3 * 255
    assert not np.array_equal(dithered, naive)

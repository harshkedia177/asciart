import numpy as np
import pytest

from asciart.core.glyph_cache import GlyphMatcher


CHARS = " .:-=+*#%@"


@pytest.fixture
def matcher():
    """A GlyphMatcher with the standard 10-char ramp."""
    return GlyphMatcher(CHARS, cell_w=6, cell_h=10)


def test_glyph_matcher_creates(matcher):
    """Can instantiate GlyphMatcher with a character set."""
    assert matcher is not None
    assert len(matcher.glyphs) == len(CHARS)
    assert matcher.cell_w == 6
    assert matcher.cell_h == 10


def test_glyph_matcher_has_features(matcher):
    """Each glyph has char, mean_brightness, and 16 features."""
    for glyph in matcher.glyphs:
        assert isinstance(glyph.char, str)
        assert isinstance(glyph.mean_brightness, float)
        assert glyph.features.shape == (16,)
        assert glyph.bitmap.shape == (10, 6)
        assert isinstance(glyph.variance, float)
        assert isinstance(glyph.frequency, float)


def test_space_brightness(matcher):
    """Space character should have low brightness (mostly black bitmap)."""
    space_glyph = matcher.glyphs[0]  # First char is space
    assert space_glyph.char == " "
    assert space_glyph.mean_brightness < 0.1


def test_match_tile_returns_valid_index(matcher):
    """match_tile returns an int in valid range."""
    tile = np.random.rand(10, 6)
    idx = matcher.match_tile(tile, mode="structural")
    assert isinstance(idx, int)
    assert 0 <= idx < len(CHARS)


def test_match_tiles_batch(matcher):
    """Batch processing returns correct shape."""
    grid_h, grid_w = 5, 8
    tiles = np.random.rand(grid_h, grid_w, 10, 6)
    result = matcher.match_tiles_batch(tiles, mode="structural")
    assert result.shape == (grid_h, grid_w)
    assert result.dtype == np.uint8
    assert np.all(result < len(CHARS))


def test_structural_mode_works(matcher):
    """Structural matching doesn't crash on random tiles."""
    tiles = np.random.rand(3, 4, 10, 6)
    result = matcher.match_tiles_batch(tiles, mode="structural")
    assert result.shape == (3, 4)
    assert result.dtype == np.uint8


def test_hybrid_mode_works(matcher):
    """Hybrid matching doesn't crash on random tiles."""
    tiles = np.random.rand(3, 4, 10, 6)
    result = matcher.match_tiles_batch(tiles, mode="hybrid")
    assert result.shape == (3, 4)
    assert result.dtype == np.uint8


def test_adaptive_weights_short_ramp():
    short = GlyphMatcher(" .#@", cell_w=6, cell_h=10)
    assert short.w_brightness == pytest.approx(0.70)
    assert short.w_structure == pytest.approx(0.20)

    standard = GlyphMatcher(CHARS, cell_w=6, cell_h=10)
    assert standard.w_brightness == pytest.approx(0.25)
    assert standard.w_structure == pytest.approx(0.50)


def test_brightness_mode_works(matcher):
    """Brightness-only matching works correctly."""
    # Create a black tile (all zeros) -> should match darkest glyph
    black_tile = np.zeros((10, 6))
    idx = matcher.match_tile(black_tile, mode="brightness")
    assert isinstance(idx, int)
    assert 0 <= idx < len(CHARS)

    # Batch version
    tiles = np.random.rand(2, 3, 10, 6)
    result = matcher.match_tiles_batch(tiles, mode="brightness")
    assert result.shape == (2, 3)
    assert result.dtype == np.uint8

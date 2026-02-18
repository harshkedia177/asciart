from __future__ import annotations
import numpy as np


def _floyd_steinberg_python(image: np.ndarray, num_levels: int) -> np.ndarray:
    """Pure-Python Floyd-Steinberg error-diffusion dithering."""
    img = image.astype(np.float64).copy()
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            level = round(old / 255.0 * (num_levels - 1))
            level = max(0, min(num_levels - 1, level))
            new = level / (num_levels - 1) * 255.0
            img[y, x] = new
            error = old - new
            if x + 1 < w:
                img[y, x + 1] += error * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += error * 3 / 16
                img[y + 1, x] += error * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 1 / 16
    return np.clip(img, 0, 255)


def floyd_steinberg(image: np.ndarray, num_levels: int) -> np.ndarray:
    """Floyd-Steinberg dithering with automatic JIT acceleration when available."""
    from asciart.accel import get_backend

    backend = get_backend()
    if backend.has_jit:
        from asciart.accel.numba_kernels import floyd_steinberg_jit

        return floyd_steinberg_jit(image.astype(np.float64).copy(), num_levels)
    return _floyd_steinberg_python(image, num_levels)


def _atkinson_python(image: np.ndarray, num_levels: int) -> np.ndarray:
    """Pure-Python Atkinson dithering -- only diffuses 75% of error (6/8).

    Creates crisper highlights and shadows than Floyd-Steinberg.
    Each of 6 neighbors receives 1/8 of the quantization error;
    the remaining 2/8 (25%) is intentionally discarded.
    """
    img = image.astype(np.float64).copy()
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            level = round(old / 255.0 * (num_levels - 1))
            level = max(0, min(num_levels - 1, level))
            new = level / (num_levels - 1) * 255.0
            img[y, x] = new
            error = old - new
            frac = error / 8.0
            # Right neighbors on same row
            if x + 1 < w:
                img[y, x + 1] += frac
            if x + 2 < w:
                img[y, x + 2] += frac
            # Next row
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += frac
                img[y + 1, x] += frac
                if x + 1 < w:
                    img[y + 1, x + 1] += frac
            # Two rows down
            if y + 2 < h:
                img[y + 2, x] += frac
    return np.clip(img, 0, 255)


def atkinson(image: np.ndarray, num_levels: int) -> np.ndarray:
    """Atkinson dithering with automatic JIT acceleration when available."""
    from asciart.accel import get_backend

    backend = get_backend()
    if backend.has_jit:
        from asciart.accel.numba_kernels import atkinson_jit

        return atkinson_jit(image.astype(np.float64).copy(), num_levels)
    return _atkinson_python(image, num_levels)


def _generate_bayer(n: int) -> np.ndarray:
    """Recursively generate an n x n Bayer threshold matrix (n must be power of 2)."""
    if n == 1:
        return np.array([[0]])
    smaller = _generate_bayer(n // 2)
    return np.block([
        [4 * smaller + 0, 4 * smaller + 2],
        [4 * smaller + 3, 4 * smaller + 1],
    ]) / (n * n)


BAYER_8X8 = _generate_bayer(8)


# Pre-computed 8x8 blue noise threshold matrix.
# Values are hand-tuned to approximate a blue-noise spectral profile,
# providing visually pleasing non-periodic dithering patterns.
BLUE_NOISE_8X8 = np.array([
    [34,  29,  17,  21,  30,   7,  11,  38],
    [ 2,  52,   9,  45,   3,  49,  27,  15],
    [43,  14,  60,  24,  56,  19,  40,  58],
    [22,  36,   5,  41,  10,  33,   1,  25],
    [54,  47,  28,  13,  62,  44,  53,  16],
    [ 8,  18,  57,  48,  23,   6,  31,  42],
    [37,  63,  35,   4,  39,  55,  20,  59],
    [26,  12,  50,  32,  51,  14,  46,   0],
], dtype=np.float64) / 64.0


def _threshold_dither(image: np.ndarray, num_levels: int, matrix: np.ndarray) -> np.ndarray:
    """Apply threshold-based dithering with a tiled matrix (Bayer, blue noise, etc.)."""
    h, w = image.shape
    mh, mw = matrix.shape
    threshold = np.tile(matrix, ((h + mh - 1) // mh, (w + mw - 1) // mw))[:h, :w]
    normalized = image / 255.0
    biased = normalized + (threshold - 0.5) / num_levels
    quantized = np.round(biased * (num_levels - 1)).clip(0, num_levels - 1)
    return quantized / (num_levels - 1) * 255.0


def ordered_dither(image: np.ndarray, num_levels: int) -> np.ndarray:
    return _threshold_dither(image, num_levels, BAYER_8X8)


def blue_noise_dither(image: np.ndarray, num_levels: int) -> np.ndarray:
    return _threshold_dither(image, num_levels, BLUE_NOISE_8X8)

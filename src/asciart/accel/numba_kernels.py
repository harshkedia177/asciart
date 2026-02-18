"""Numba JIT-compiled dithering kernels.

These are identical algorithms to the pure-Python versions in
``asciart.core.dither`` but decorated with ``@njit`` for faster execution.
When Numba is not installed, stub functions that raise ``ImportError`` are
provided instead so that the module can always be imported safely.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    @njit(cache=True)
    def floyd_steinberg_jit(image: np.ndarray, num_levels: int) -> np.ndarray:
        """Floyd-Steinberg error-diffusion dithering (JIT-compiled)."""
        img = image.copy()
        h, w = img.shape
        for y in range(h):
            for x in range(w):
                old = img[y, x]
                level = round(old / 255.0 * (num_levels - 1))
                if level < 0:
                    level = 0
                if level > num_levels - 1:
                    level = num_levels - 1
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
        # Clip to [0, 255]
        for y in range(h):
            for x in range(w):
                val = img[y, x]
                if val < 0:
                    img[y, x] = 0.0
                elif val > 255:
                    img[y, x] = 255.0
        return img

    @njit(cache=True)
    def atkinson_jit(image: np.ndarray, num_levels: int) -> np.ndarray:
        """Atkinson error-diffusion dithering (JIT-compiled).

        Only diffuses 75% of error (6/8).  Creates crisper highlights
        and shadows than Floyd-Steinberg.
        """
        img = image.copy()
        h, w = img.shape
        for y in range(h):
            for x in range(w):
                old = img[y, x]
                level = round(old / 255.0 * (num_levels - 1))
                if level < 0:
                    level = 0
                if level > num_levels - 1:
                    level = num_levels - 1
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
        # Clip to [0, 255]
        for y in range(h):
            for x in range(w):
                val = img[y, x]
                if val < 0:
                    img[y, x] = 0.0
                elif val > 255:
                    img[y, x] = 255.0
        return img

except ImportError:

    def floyd_steinberg_jit(image, num_levels):  # type: ignore[misc]
        raise ImportError("numba is required for JIT kernels")

    def atkinson_jit(image, num_levels):  # type: ignore[misc]
        raise ImportError("numba is required for JIT kernels")

# src/asciart/core/dither.py
from __future__ import annotations
import numpy as np


def floyd_steinberg(image: np.ndarray, num_levels: int) -> np.ndarray:
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


BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float64) / 16.0


def ordered_dither(image: np.ndarray, num_levels: int) -> np.ndarray:
    h, w = image.shape
    mh, mw = BAYER_4X4.shape
    threshold = np.tile(BAYER_4X4, ((h + mh - 1) // mh, (w + mw - 1) // mw))[:h, :w]
    normalized = image / 255.0
    biased = normalized + (threshold - 0.5) / num_levels
    quantized = np.round(biased * (num_levels - 1)).clip(0, num_levels - 1)
    return quantized / (num_levels - 1) * 255.0

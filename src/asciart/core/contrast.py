# src/asciart/core/contrast.py
from __future__ import annotations

import numpy as np


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to a grayscale image.

    Divides image into grid_size x grid_size tiles, equalizes each independently
    with a contrast limit to prevent noise amplification.

    Args:
        gray: 2D float array (H, W) with values 0-255
        clip_limit: Multiplier for histogram clipping
        grid_size: Number of tiles along each axis
    Returns:
        Enhanced grayscale array, same shape, values 0-255
    """
    h, w = gray.shape
    # Clamp to valid range and convert to uint8 for histogram binning
    clamped = np.clip(gray, 0.0, 255.0)

    # Compute tile boundaries
    tile_h = h / grid_size
    tile_w = w / grid_size

    # Build a CDF lookup for each tile
    # cdfs shape: (grid_size, grid_size, 256)
    cdfs = np.zeros((grid_size, grid_size, 256), dtype=np.float64)

    for ty in range(grid_size):
        y0 = int(round(ty * tile_h))
        y1 = int(round((ty + 1) * tile_h))
        y1 = max(y1, y0 + 1)  # ensure at least 1 row
        y1 = min(y1, h)
        for tx in range(grid_size):
            x0 = int(round(tx * tile_w))
            x1 = int(round((tx + 1) * tile_w))
            x1 = max(x1, x0 + 1)  # ensure at least 1 col
            x1 = min(x1, w)

            tile = clamped[y0:y1, x0:x1]
            cdfs[ty, tx] = _tile_cdf(tile, clip_limit)

    # Map each pixel through its tile's CDF
    result = np.empty_like(gray, dtype=np.float64)
    pixel_vals = np.clip(np.round(clamped), 0, 255).astype(np.intp)

    for ty in range(grid_size):
        y0 = int(round(ty * tile_h))
        y1 = int(round((ty + 1) * tile_h))
        y1 = max(y1, y0 + 1)
        y1 = min(y1, h)
        for tx in range(grid_size):
            x0 = int(round(tx * tile_w))
            x1 = int(round((tx + 1) * tile_w))
            x1 = max(x1, x0 + 1)
            x1 = min(x1, w)

            tile_indices = pixel_vals[y0:y1, x0:x1]
            result[y0:y1, x0:x1] = cdfs[ty, tx][tile_indices]

    return result


def _tile_cdf(tile: np.ndarray, clip_limit: float) -> np.ndarray:
    """Compute clipped, redistributed CDF for a single tile.

    Returns a 256-element lookup table mapping pixel value -> equalized value.
    """
    # Flatten and compute histogram
    flat = np.clip(np.round(tile.ravel()), 0, 255).astype(np.intp)
    hist = np.bincount(flat, minlength=256).astype(np.float64)

    n_pixels = flat.size
    if n_pixels == 0:
        return np.arange(256, dtype=np.float64)

    # Clip histogram
    mean_count = n_pixels / 256.0
    limit = clip_limit * mean_count

    excess = 0.0
    for i in range(256):
        if hist[i] > limit:
            excess += hist[i] - limit
            hist[i] = limit

    # Redistribute excess evenly
    redistribute = excess / 256.0
    hist += redistribute

    # Compute CDF
    cdf = np.cumsum(hist)

    # Normalize CDF to 0-255
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0.0
    cdf_max = cdf[-1]
    denom = cdf_max - cdf_min
    if denom == 0:
        return np.arange(256, dtype=np.float64)

    normalized = (cdf - cdf_min) / denom * 255.0
    return np.clip(normalized, 0.0, 255.0)

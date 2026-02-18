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
    clamped = np.clip(gray, 0.0, 255.0)

    tile_h = h / grid_size
    tile_w = w / grid_size
    bounds = _tile_bounds(grid_size, tile_h, tile_w, h, w)

    # Build a CDF lookup for each tile
    cdfs = np.zeros((grid_size, grid_size, 256), dtype=np.float64)
    for ty in range(grid_size):
        for tx in range(grid_size):
            y0, y1, x0, x1 = bounds[ty][tx]
            cdfs[ty, tx] = _tile_cdf(clamped[y0:y1, x0:x1], clip_limit)

    # Map each pixel through its tile's CDF
    result = np.empty_like(gray, dtype=np.float64)
    pixel_vals = np.clip(np.round(clamped), 0, 255).astype(np.intp)

    for ty in range(grid_size):
        for tx in range(grid_size):
            y0, y1, x0, x1 = bounds[ty][tx]
            result[y0:y1, x0:x1] = cdfs[ty, tx][pixel_vals[y0:y1, x0:x1]]

    return result


def _tile_bounds(
    grid_size: int, tile_h: float, tile_w: float, h: int, w: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Compute (y0, y1, x0, x1) bounds for each tile in the grid."""
    bounds: list[list[tuple[int, int, int, int]]] = []
    for ty in range(grid_size):
        row: list[tuple[int, int, int, int]] = []
        y0 = int(round(ty * tile_h))
        y1 = min(max(int(round((ty + 1) * tile_h)), y0 + 1), h)
        for tx in range(grid_size):
            x0 = int(round(tx * tile_w))
            x1 = min(max(int(round((tx + 1) * tile_w)), x0 + 1), w)
            row.append((y0, y1, x0, x1))
        bounds.append(row)
    return bounds


def _tile_cdf(tile: np.ndarray, clip_limit: float) -> np.ndarray:
    """Compute clipped, redistributed CDF for a single tile.

    Returns a 256-element lookup table mapping pixel value -> equalized value.
    """
    flat = np.clip(np.round(tile.ravel()), 0, 255).astype(np.intp)
    hist = np.bincount(flat, minlength=256).astype(np.float64)

    n_pixels = flat.size
    if n_pixels == 0:
        return np.arange(256, dtype=np.float64)

    # Clip histogram and redistribute excess evenly
    limit = clip_limit * (n_pixels / 256.0)
    over = np.maximum(hist - limit, 0.0)
    hist = np.minimum(hist, limit) + over.sum() / 256.0

    # Compute and normalize CDF to 0-255
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0.0
    denom = cdf[-1] - cdf_min
    if denom == 0:
        return np.arange(256, dtype=np.float64)

    return np.clip((cdf - cdf_min) / denom * 255.0, 0.0, 255.0)

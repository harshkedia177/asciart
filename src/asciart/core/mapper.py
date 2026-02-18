# src/asciart/core/mapper.py
from __future__ import annotations

import numpy as np


def map_brightness(
    gray: np.ndarray,
    chars: str,
    invert: bool,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Map grayscale values to character indices from a ramp (fully vectorized).

    Args:
        gray: 2D float array (H, W) with values 0-255.
        chars: Character ramp string, ordered light-to-dark.
        invert: If True, reverse the ramp.
        colors: Optional RGB array (H, W, 3) for color attachment.

    Returns:
        (char_indices, char_map, fg_colors)
        - char_indices: (H, W) uint8 array of indices into char_map
        - char_map: the ramp string (possibly inverted) to use
        - fg_colors: (H, W, 3) uint8 or None
    """
    ramp = chars[::-1] if invert else chars
    num_chars = len(ramp)

    # Bright pixels (high gray=255) -> low index (light chars like space)
    # Dark pixels (low gray=0) -> high index (dark chars like @/#)
    char_indices = np.clip(
        ((1.0 - gray / 255.0) * (num_chars - 1)).astype(np.int32),
        0,
        num_chars - 1,
    ).astype(np.uint8)

    fg = None
    if colors is not None:
        fg = np.clip(colors, 0, 255).astype(np.uint8)

    return char_indices, ramp, fg


def map_braille(
    gray: np.ndarray,
    threshold: float = 128.0,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Map grayscale image to braille character indices (fully vectorized).

    Each braille char encodes a 2x4 pixel block. The image dimensions should
    already account for this (width = output_cols * 2, height = output_rows * 4).

    Returns:
        (char_indices, char_map, fg_colors)
        - char_indices: (H//4, W//2) uint8 — offsets from U+2800
        - char_map: string of 256 braille chars (chr(0x2800) to chr(0x28FF))
        - fg_colors: (H//4, W//2, 3) uint8 or None
    """
    h, w = gray.shape

    # Pad to multiples of 4 (height) and 2 (width)
    pad_h = (4 - h % 4) % 4
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
        if colors is not None:
            colors = np.pad(colors, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)

    h, w = gray.shape
    rows_out = h // 4
    cols_out = w // 2

    # Build the canonical braille char_map: 256 chars from U+2800 to U+28FF
    char_map = "".join(chr(0x2800 + i) for i in range(256))

    # Binary mask: dark pixels (below threshold) become dots ON
    binary = (gray < threshold).astype(np.uint8)

    # Reshape to (rows_out, 4, cols_out, 2) for block-level access
    blocks = binary.reshape(rows_out, 4, cols_out, 2)

    # Braille dot bit layout for a 2x4 block:
    #   (row=0, col=0) -> bit 0 (0x01)
    #   (row=1, col=0) -> bit 1 (0x02)
    #   (row=2, col=0) -> bit 2 (0x04)
    #   (row=0, col=1) -> bit 3 (0x08)
    #   (row=1, col=1) -> bit 4 (0x10)
    #   (row=2, col=1) -> bit 5 (0x20)
    #   (row=3, col=0) -> bit 6 (0x40)
    #   (row=3, col=1) -> bit 7 (0x80)
    #
    # Compute each bit plane and combine with bitwise OR.
    char_indices = (
        blocks[:, 0, :, 0].astype(np.uint16) * 0x01
        | blocks[:, 1, :, 0].astype(np.uint16) * 0x02
        | blocks[:, 2, :, 0].astype(np.uint16) * 0x04
        | blocks[:, 0, :, 1].astype(np.uint16) * 0x08
        | blocks[:, 1, :, 1].astype(np.uint16) * 0x10
        | blocks[:, 2, :, 1].astype(np.uint16) * 0x20
        | blocks[:, 3, :, 0].astype(np.uint16) * 0x40
        | blocks[:, 3, :, 1].astype(np.uint16) * 0x80
    ).astype(np.uint8)

    fg = None
    if colors is not None:
        # Average color across each 4x2 block -> (rows_out, cols_out, 3)
        color_blocks = colors.reshape(rows_out, 4, cols_out, 2, 3).astype(np.float64)
        fg = np.clip(color_blocks.mean(axis=(1, 3)), 0, 255).astype(np.uint8)

    return char_indices, char_map, fg


def map_halfblock(
    pixels: np.ndarray,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """Map pixels to half-block characters with fg/bg color (fully vectorized).

    Each character cell encodes two vertically stacked pixels using upper half
    block. FG = top pixel color, BG = bottom pixel color.

    Returns:
        (char_indices, char_map, fg, bg)
        - char_indices: (H//2, W) uint8 — all zeros (single char map)
        - char_map: "\u2580"
        - fg: (H//2, W, 3) uint8 — top row pixel colors
        - bg: (H//2, W, 3) uint8 — bottom row pixel colors
    """
    h, w = pixels.shape[:2]
    if h % 2 != 0:
        pixels = np.pad(pixels, ((0, 1), (0, 0), (0, 0)), mode="edge")
        h += 1

    char_map = "\u2580"

    # Top rows: even indices (0, 2, 4, ...)
    # Bottom rows: odd indices (1, 3, 5, ...)
    fg = np.clip(pixels[0::2], 0, 255).astype(np.uint8)
    bg = np.clip(pixels[1::2], 0, 255).astype(np.uint8)

    out_h = h // 2
    char_indices = np.zeros((out_h, w), dtype=np.uint8)

    return char_indices, char_map, fg, bg

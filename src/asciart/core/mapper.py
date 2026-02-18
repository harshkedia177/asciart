# src/asciart/core/mapper.py
from __future__ import annotations

import numpy as np
from asciart.models import Cell


def map_brightness(gray: np.ndarray, chars: str, invert: bool, colors: np.ndarray | None = None) -> list[list[Cell]]:
    """Map grayscale values to characters from a ramp.

    Args:
        gray: 2D float array (H, W) with values 0-255.
        chars: Character ramp string, ordered light-to-dark.
        invert: If True, reverse the ramp.
        colors: Optional RGB array (H, W, 3) for color attachment.

    Returns:
        2D list of Cells.
    """
    ramp = chars[::-1] if invert else chars
    num_chars = len(ramp)

    # Bright pixels (high gray) -> low index (light chars), dark pixels (low gray) -> high index (dark chars)
    indices = np.clip(((1.0 - gray / 255.0) * (num_chars - 1)).astype(int), 0, num_chars - 1)

    rows = []
    h, w = gray.shape
    for y in range(h):
        row = []
        for x in range(w):
            char = ramp[indices[y, x]]
            fg = None
            if colors is not None:
                fg = (int(colors[y, x, 0]), int(colors[y, x, 1]), int(colors[y, x, 2]))
            row.append(Cell(char=char, fg=fg))
        rows.append(row)

    return rows


def map_braille(gray: np.ndarray, threshold: float = 128.0, colors: np.ndarray | None = None) -> list[list[Cell]]:
    """Map grayscale image to braille characters.

    Each braille char encodes a 2x4 pixel block. The image dimensions should already
    account for this (width = output_cols * 2, height = output_rows * 4).
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

    # Braille dot positions: (row_offset, col_offset, bit_value)
    dot_map = [
        (0, 0, 0x01), (1, 0, 0x02), (2, 0, 0x04),
        (0, 1, 0x08), (1, 1, 0x10), (2, 1, 0x20),
        (3, 0, 0x40), (3, 1, 0x80),
    ]

    rows = []
    for by in range(rows_out):
        row = []
        for bx in range(cols_out):
            y0 = by * 4
            x0 = bx * 2
            offset = 0
            for dy, dx, bit in dot_map:
                if gray[y0 + dy, x0 + dx] < threshold:
                    offset |= bit

            fg = None
            if colors is not None:
                block = colors[y0:y0 + 4, x0:x0 + 2]
                avg = block.mean(axis=(0, 1))
                fg = (int(avg[0]), int(avg[1]), int(avg[2]))

            row.append(Cell(char=chr(0x2800 + offset), fg=fg))
        rows.append(row)

    return rows


def map_halfblock(pixels: np.ndarray) -> list[list[Cell]]:
    """Map pixels to half-block characters with fg/bg color.

    Each character cell encodes two vertically stacked pixels using \u2580 (upper half block).
    FG = top pixel color, BG = bottom pixel color. Doubles vertical resolution.
    """
    h, w, _ = pixels.shape
    if h % 2 != 0:
        pixels = np.pad(pixels, ((0, 1), (0, 0), (0, 0)), mode="edge")
        h += 1

    rows = []
    for y in range(0, h, 2):
        row = []
        for x in range(w):
            top = pixels[y, x]
            bottom = pixels[y + 1, x]
            fg = (int(top[0]), int(top[1]), int(top[2]))
            bg = (int(bottom[0]), int(bottom[1]), int(bottom[2]))
            row.append(Cell(char="\u2580", fg=fg, bg=bg))
        rows.append(row)

    return rows

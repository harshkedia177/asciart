# src/asciart/core/engine.py
from __future__ import annotations

import numpy as np
from PIL import Image

from asciart.models import AsciiArt, Cell, ConvertOptions, ColorMode, DitherMode, Mode
from asciart.core.preprocess import preprocess, resize, to_grayscale
from asciart.core.mapper import map_brightness, map_braille, map_halfblock
from asciart.core.ramps import BLOCKS
from asciart.core.dither import floyd_steinberg, ordered_dither
from asciart.core.edges import detect_edges, angle_to_char


def convert(image: Image.Image, options: ConvertOptions) -> AsciiArt:
    """Main conversion pipeline: Image + Options -> AsciiArt grid."""
    # 1. Preprocess
    img = preprocess(image, options.brightness, options.contrast, options.saturation, options.sharpness)

    # 2. Determine if we need color
    need_color = options.color not in (ColorMode.NONE,)
    if options.color == ColorMode.AUTO:
        need_color = True

    # 3. Mode-specific sizing and mapping
    if options.mode == Mode.BRAILLE:
        # Braille: each char = 2x4 pixels, so resize to 2x width and ~4x height
        target_w = options.width * 2
        img_w, img_h = img.size
        target_h = options.height
        if target_h is None:
            target_h = int((img_h / img_w) * target_w * options.font_ratio)
            target_h = max(4, ((target_h + 3) // 4) * 4)

        img = img.resize((target_w, target_h), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        gray = to_grayscale(pixels)
        colors = pixels if need_color else None
        if options.invert:
            gray = 255.0 - gray
        cells = map_braille(gray, 128.0, colors)

    elif options.mode == Mode.HALFBLOCK:
        # Halfblock: each char = 1x2 pixels
        target_w = options.width
        img_w, img_h = img.size
        target_h = options.height
        if target_h is None:
            target_h = int((img_h / img_w) * target_w * options.font_ratio * 2)
            target_h = max(2, target_h)
            if target_h % 2 != 0:
                target_h += 1

        img = img.resize((target_w, target_h), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        cells = map_halfblock(pixels)

    else:
        # ASCII and BLOCKS modes
        img = resize(img, options.width, options.height, options.font_ratio)
        pixels = np.array(img, dtype=np.float64)
        gray = to_grayscale(pixels)
        colors = pixels if need_color else None

        # Save pre-dither grayscale for edge detection
        gray_for_edges = gray.copy() if options.edge_detection else None

        if options.mode == Mode.BLOCKS:
            chars = BLOCKS
        else:
            chars = options.chars

        # Dithering
        if options.dither == DitherMode.FLOYD_STEINBERG:
            gray = floyd_steinberg(gray, len(chars))
        elif options.dither == DitherMode.ORDERED:
            gray = ordered_dither(gray, len(chars))

        cells = map_brightness(gray, chars, options.invert, colors)

        # Edge detection on pre-dither grayscale
        if options.edge_detection and gray_for_edges is not None:
            edge_mask, edge_angles = detect_edges(gray_for_edges, options.edge_threshold)
            for y in range(len(cells)):
                for x in range(len(cells[y])):
                    if edge_mask[y, x]:
                        cells[y][x] = Cell(char=angle_to_char(edge_angles[y, x]), fg=cells[y][x].fg)

    out_h = len(cells)
    out_w = len(cells[0]) if cells else 0
    return AsciiArt(width=out_w, height=out_h, cells=cells)

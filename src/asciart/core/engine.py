# src/asciart/core/engine.py
from __future__ import annotations

import numpy as np
from PIL import Image

from asciart.accel import get_backend
from asciart.models import AsciiArt, ConvertOptions, ColorMode, DitherMode, MatchMode, Mode
from asciart.core.preprocess import preprocess, resize_with_edge_preservation, to_grayscale
from asciart.core.mapper import map_brightness, map_braille, map_halfblock
from asciart.core.ramps import BLOCKS
from asciart.core.dither import floyd_steinberg, ordered_dither, atkinson, blue_noise_dither
from asciart.core.edges import detect_edges


def convert(image: Image.Image, options: ConvertOptions) -> AsciiArt:
    """Main conversion pipeline: Image + Options -> AsciiArt grid."""
    # Detect compute backend (numpy, numba, or cupy)
    backend = get_backend()
    xp = backend.xp

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
        pixels = xp.asarray(np.array(img, dtype=np.float64))
        gray = to_grayscale(pixels, xp=xp)
        colors = pixels if need_color else None
        if options.invert:
            gray = 255.0 - gray

        # Transfer back to CPU for mapper (operates on numpy arrays)
        if backend.has_gpu:
            gray = xp.asnumpy(gray)
            if colors is not None:
                colors = xp.asnumpy(colors)

        char_indices, char_map, fg = map_braille(gray, 128.0, colors)
        return AsciiArt.from_arrays(char_indices, char_map, fg_array=fg)

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
        pixels = xp.asarray(np.array(img, dtype=np.float64))

        # Transfer back to CPU for mapper
        if backend.has_gpu:
            pixels = xp.asnumpy(pixels)

        char_indices, char_map, fg, bg = map_halfblock(pixels)
        return AsciiArt.from_arrays(char_indices, char_map, fg_array=fg, bg_array=bg)

    else:
        # ASCII and BLOCKS modes
        img = resize_with_edge_preservation(img, options.width, options.height, options.font_ratio)
        pixels = xp.asarray(np.array(img, dtype=np.float64))
        gray = to_grayscale(pixels, xp=xp)
        colors = pixels if need_color else None

        # Transfer back to CPU for downstream operations that require numpy
        if backend.has_gpu:
            gray = xp.asnumpy(gray)
            if colors is not None:
                colors = xp.asnumpy(colors)

        # Apply CLAHE if requested
        if options.clahe:
            from asciart.core.contrast import apply_clahe
            gray = apply_clahe(gray)

        # Save pre-dither grayscale for edge detection
        gray_for_edges = gray.copy() if options.edge_detection else None

        if options.mode == Mode.BLOCKS:
            chars = BLOCKS
        else:
            chars = options.chars

        # Structural / hybrid glyph matching path
        if options.match_mode != MatchMode.BRIGHTNESS:
            from asciart.core.glyph_cache import GlyphMatcher

            cell_w, cell_h = 6, 10
            target_h, target_w = gray.shape
            matcher = GlyphMatcher(
                chars, cell_w=cell_w, cell_h=cell_h,
                font_path=options.font_path,
            )

            # Resize to exact tile grid dimensions
            pixel_h = target_h * cell_h
            pixel_w = target_w * cell_w
            img_hires = img.resize((pixel_w, pixel_h), Image.LANCZOS)
            pixels_hires = np.array(img_hires, dtype=np.float64)
            gray_hires = to_grayscale(pixels_hires) / 255.0

            if options.invert:
                gray_hires = 1.0 - gray_hires

            # Split into tiles: (target_h, cell_h, target_w, cell_w) -> (target_h, target_w, cell_h, cell_w)
            tiles = gray_hires.reshape(target_h, cell_h, target_w, cell_w).transpose(0, 2, 1, 3)

            mode_str = options.match_mode.value
            char_indices = matcher.match_tiles_batch(tiles, mode=mode_str)
            char_map = chars

            # Extract colors from high-res image (average per tile)
            fg = None
            if colors is not None:
                colors_hires = np.clip(pixels_hires, 0, 255).astype(np.uint8)
                # Reshape to (target_h, cell_h, target_w, cell_w, 3) then average over cell dims
                color_tiles = colors_hires.reshape(
                    target_h, cell_h, target_w, cell_w, 3
                ).transpose(0, 2, 1, 3, 4)
                fg = color_tiles.mean(axis=(2, 3)).astype(np.uint8)

            return AsciiArt.from_arrays(char_indices, char_map, fg_array=fg)

        # Dithering
        if options.dither == DitherMode.FLOYD_STEINBERG:
            gray = floyd_steinberg(gray, len(chars))
        elif options.dither == DitherMode.ORDERED:
            gray = ordered_dither(gray, len(chars))
        elif options.dither == DitherMode.ATKINSON:
            gray = atkinson(gray, len(chars))
        elif options.dither == DitherMode.BLUE_NOISE:
            gray = blue_noise_dither(gray, len(chars))

        char_indices, char_map, fg = map_brightness(gray, chars, options.invert, colors)

        # Edge detection overlay on pre-dither grayscale
        if options.edge_detection and gray_for_edges is not None:
            edge_mask, edge_angles = detect_edges(gray_for_edges, options.edge_threshold)

            # Vectorized angle_to_char: classify angles into edge chars
            a = edge_angles % np.pi
            # Classify angles into edge character indices
            # Classify: "-" for a < pi/8 or a > 7*pi/8
            #           "/" for pi/8 <= a < 3*pi/8
            #           "|" for 3*pi/8 <= a < 5*pi/8
            #           "\" for 5*pi/8 <= a <= 7*pi/8
            edge_chars = "-/|\\"
            # Build the extended char_map: original ramp + edge chars (deduped)
            extended_map = char_map
            edge_index_map = {}
            for ec in edge_chars:
                pos = extended_map.find(ec)
                if pos == -1:
                    edge_index_map[ec] = len(extended_map)
                    extended_map += ec
                else:
                    edge_index_map[ec] = pos

            # Create index arrays for each edge char classification
            idx_dash = np.uint8(edge_index_map["-"])
            idx_slash = np.uint8(edge_index_map["/"])
            idx_pipe = np.uint8(edge_index_map["|"])
            idx_bslash = np.uint8(edge_index_map["\\"])

            # Vectorized classification using numpy where chains
            edge_indices = np.where(
                (a < np.pi / 8) | (a > 7 * np.pi / 8),
                idx_dash,
                np.where(
                    a < 3 * np.pi / 8,
                    idx_slash,
                    np.where(
                        a < 5 * np.pi / 8,
                        idx_pipe,
                        idx_bslash,
                    ),
                ),
            ).astype(np.uint8)

            # Apply edge overlay: where edge_mask is True, replace char_indices
            char_indices = np.where(edge_mask, edge_indices, char_indices).astype(np.uint8)
            char_map = extended_map

        return AsciiArt.from_arrays(char_indices, char_map, fg_array=fg)

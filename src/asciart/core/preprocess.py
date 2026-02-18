from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def preprocess(image: Image.Image, brightness: float, contrast: float, saturation: float, sharpness: float) -> Image.Image:
    """Apply image adjustments. All values are multipliers (1.0 = no change) except brightness which is an offset."""
    img = image.convert("RGB")

    if brightness != 0.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + brightness / 100.0)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)

    return img


def resize(image: Image.Image, width: int, height: int | None, font_ratio: float) -> Image.Image:
    """Resize image to target character dimensions with aspect ratio correction."""
    img_w, img_h = image.size
    if height is None:
        height = int((img_h / img_w) * width * font_ratio)
        height = max(1, height)
    return image.resize((width, height), Image.LANCZOS)


def resize_with_edge_preservation(
    image: Image.Image,
    width: int,
    height: int | None,
    font_ratio: float,
) -> Image.Image:
    """Resize with pre-sharpening to preserve edges during heavy downscaling.

    When the horizontal scale factor exceeds 2x, an UnsharpMask is applied
    before the LANCZOS resize so that fine edges survive the downscale.
    """
    img_w, img_h = image.size
    if height is None:
        height = int((img_h / img_w) * width * font_ratio)
        height = max(1, height)
    scale_factor = img_w / width
    if scale_factor > 2.0:
        image = image.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
    return image.resize((width, height), Image.LANCZOS)


def to_grayscale(pixels: np.ndarray, xp=None) -> np.ndarray:
    """Convert RGB pixel array to grayscale using BT.709 weights. Returns float64 array [0, 255].

    Parameters
    ----------
    pixels:
        H x W x 3 array (numpy or cupy).
    xp:
        Array module to use (numpy or cupy).  When *None*, numpy is used.
    """
    if xp is None:
        xp = np
    return pixels[:, :, 0] * 0.2126 + pixels[:, :, 1] * 0.7152 + pixels[:, :, 2] * 0.0722

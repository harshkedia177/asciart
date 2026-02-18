from __future__ import annotations

from PIL import Image

from asciart.core.engine import convert
from asciart.models import AsciiArt, ConvertOptions


def extract_frames(path: str) -> tuple[list[Image.Image], list[int]]:
    """Extract all frames and durations from an animated GIF.

    Returns:
        Tuple of (frames as RGB PIL Images, durations in ms).
    """
    img = Image.open(path)
    frames = []
    durations = []

    try:
        while True:
            frames.append(img.convert("RGB").copy())
            durations.append(img.info.get("duration", 100))
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    return frames, durations


def convert_gif(path: str, options: ConvertOptions) -> list[AsciiArt]:
    """Convert all frames of an animated GIF to AsciiArt."""
    frames, _ = extract_frames(path)
    return [convert(frame, options) for frame in frames]

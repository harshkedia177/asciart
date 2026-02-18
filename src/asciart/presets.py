from __future__ import annotations

from asciart.models import ConvertOptions, Mode, ColorMode, DitherMode, MatchMode

PRESETS: dict[str, ConvertOptions] = {
    "photo": ConvertOptions(
        mode=Mode.ASCII,
        chars=" .:-=+*#%@",
        color=ColorMode.TRUECOLOR,
        dither=DitherMode.FLOYD_STEINBERG,
        contrast=1.2,
    ),
    "logo": ConvertOptions(
        mode=Mode.BRAILLE,
        color=ColorMode.TRUECOLOR,
        contrast=1.5,
        edge_detection=True,
    ),
    "retro": ConvertOptions(
        mode=Mode.ASCII,
        chars=" .:#",
        color=ColorMode.NONE,
        dither=DitherMode.ORDERED,
    ),
    "hd": ConvertOptions(
        mode=Mode.BRAILLE,
        color=ColorMode.TRUECOLOR,
        width=200,
        dither=DitherMode.FLOYD_STEINBERG,
    ),
    "blocks": ConvertOptions(
        mode=Mode.HALFBLOCK,
        color=ColorMode.TRUECOLOR,
    ),
    "lineart": ConvertOptions(
        mode=Mode.ASCII,
        edge_detection=True,
        color=ColorMode.NONE,
        contrast=1.8,
    ),
    "studio": ConvertOptions(
        mode=Mode.ASCII,
        match_mode=MatchMode.HYBRID,
        clahe=True,
        dither=DitherMode.ATKINSON,
        color=ColorMode.TRUECOLOR,
        contrast=1.1,
        width=120,
    ),
}


def get_preset(name: str) -> ConvertOptions | None:
    """Get a preset by name, or None if not found."""
    return PRESETS.get(name)

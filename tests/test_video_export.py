from __future__ import annotations

import numpy as np
import pytest

from PIL import Image

from asciart.models import AsciiArt
from asciart.renderers.video_export import export_to_gif, check_ffmpeg


def _make_ascii_art(width: int = 10, height: int = 5, seed: int = 0) -> AsciiArt:
    char_array = np.full((height, width), seed % 4, dtype=np.uint8)
    fg_val = 100 + (seed * 30) % 156  # vary color per frame to avoid GIF dedup
    fg_array = np.full((height, width, 3), fg_val, dtype=np.uint8)
    return AsciiArt.from_arrays(char_array, " .:#", fg_array=fg_array)


def test_export_gif_creates_file(tmp_path):
    arts = [_make_ascii_art() for _ in range(3)]
    output = tmp_path / "test.gif"
    export_to_gif(arts, str(output), fps=10.0)
    assert output.exists()
    assert output.stat().st_size > 0
    img = Image.open(output)
    assert img.format == "GIF"


def test_export_gif_frame_count(tmp_path):
    arts = [_make_ascii_art(seed=i) for i in range(5)]
    output = tmp_path / "test.gif"
    export_to_gif(arts, str(output), fps=10.0)
    img = Image.open(output)
    frame_count = 0
    try:
        while True:
            frame_count += 1
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    assert frame_count == 5


def test_export_gif_empty_raises():
    with pytest.raises(ValueError, match="No frames"):
        export_to_gif([], "/tmp/empty.gif", fps=10.0)


def test_check_ffmpeg_returns_bool():
    result = check_ffmpeg()
    assert isinstance(result, bool)

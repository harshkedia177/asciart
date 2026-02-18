import tempfile
from pathlib import Path
from PIL import Image
from asciart.core.gif import extract_frames, convert_gif
from asciart.models import ConvertOptions, ColorMode


def _make_test_gif() -> Path:
    """Create a simple 3-frame animated GIF."""
    frames = []
    for i in range(3):
        img = Image.new("RGB", (50, 50), (i * 80, i * 80, i * 80))
        frames.append(img)

    path = Path(tempfile.mktemp(suffix=".gif"))
    frames[0].save(
        str(path), save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return path


def test_extract_frames():
    path = _make_test_gif()
    frames, durations = extract_frames(str(path))
    assert len(frames) == 3
    assert len(durations) == 3
    assert all(isinstance(f, Image.Image) for f in frames)


def test_convert_gif_returns_list():
    path = _make_test_gif()
    opts = ConvertOptions(width=10, color=ColorMode.NONE)
    arts = convert_gif(str(path), opts)
    assert len(arts) == 3
    assert all(a.width == 10 for a in arts)

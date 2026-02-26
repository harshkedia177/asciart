from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pathlib import Path  # noqa: E402

from asciart.core.video import VideoProcessor  # noqa: E402
from asciart.models import AsciiArt, ColorMode, ConvertOptions, Mode  # noqa: E402


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """Create a 5-frame synthetic video file."""
    video_path = tmp_path / "test_video.avi"
    h, w = 60, 80
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
    for i in range(5):
        frame = np.full((h, w, 3), i * 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


def test_end_to_end_video_to_ascii(synthetic_video):
    options = ConvertOptions(width=20, color=ColorMode.NONE)
    proc = VideoProcessor(
        source=str(synthetic_video),
        options=options,
        target_fps=10.0,
        temporal_smoothing=0.0,
    )
    frames = list(proc.iter_frames())
    assert len(frames) == 5
    assert all(isinstance(f, AsciiArt) for f in frames)
    assert all(f.width == 20 for f in frames)


def test_end_to_end_video_to_gif(synthetic_video, tmp_path):
    options = ConvertOptions(width=10, color=ColorMode.NONE)
    proc = VideoProcessor(
        source=str(synthetic_video),
        options=options,
        target_fps=10.0,
        temporal_smoothing=0.3,
    )
    output = tmp_path / "output.gif"
    proc.export_gif(str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_end_to_end_with_color(synthetic_video):
    options = ConvertOptions(width=20, color=ColorMode.TRUECOLOR)
    proc = VideoProcessor(
        source=str(synthetic_video),
        options=options,
        target_fps=10.0,
        temporal_smoothing=0.3,
    )
    frames = list(proc.iter_frames())
    assert len(frames) == 5
    assert all(f.fg_array is not None for f in frames)


def test_end_to_end_with_different_modes(synthetic_video):
    for mode in [Mode.ASCII, Mode.BLOCKS, Mode.BRAILLE, Mode.HALFBLOCK]:
        options = ConvertOptions(width=20, mode=mode, color=ColorMode.NONE)
        proc = VideoProcessor(
            source=str(synthetic_video),
            options=options,
            target_fps=10.0,
            temporal_smoothing=0.0,
        )
        frames = list(proc.iter_frames())
        assert len(frames) == 5, f"Mode {mode} failed"

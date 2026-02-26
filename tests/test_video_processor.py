from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unittest.mock import patch
from asciart.core.video import VideoProcessor
from asciart.models import ConvertOptions, ColorMode


def _make_bgr_frame(h: int = 60, w: int = 80, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 30.0):
        self._frames = frames
        self._index = 0
        self._fps = fps

    def isOpened(self):
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frames[0].shape[1]) if self._frames else 0.0
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frames[0].shape[0]) if self._frames else 0.0
        return 0.0

    def release(self):
        pass


def test_processor_converts_frames():
    frames = [_make_bgr_frame() for _ in range(3)]
    fake_cap = FakeCapture(frames, fps=10.0)
    options = ConvertOptions(width=20, color=ColorMode.NONE)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        proc = VideoProcessor(
            source="test.mp4",
            options=options,
            target_fps=10.0,
            temporal_smoothing=0.0,
        )
        converted = list(proc.iter_frames())
    assert len(converted) == 3
    assert all(art.width == 20 for art in converted)


def test_processor_applies_temporal_smoothing():
    frame1 = np.full((60, 80, 3), 0, dtype=np.uint8)
    frame2 = np.full((60, 80, 3), 255, dtype=np.uint8)
    fake_cap = FakeCapture([frame1, frame2], fps=10.0)
    options = ConvertOptions(width=20, color=ColorMode.NONE)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        proc = VideoProcessor(
            source="test.mp4",
            options=options,
            target_fps=10.0,
            temporal_smoothing=0.5,
        )
        converted = list(proc.iter_frames())
    assert len(converted) == 2


def test_processor_no_smoothing_when_alpha_one():
    frames = [_make_bgr_frame(value=i * 50) for i in range(3)]
    fake_cap = FakeCapture(frames, fps=10.0)
    options = ConvertOptions(width=20, color=ColorMode.NONE)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        proc = VideoProcessor(
            source="test.mp4",
            options=options,
            target_fps=10.0,
            temporal_smoothing=1.0,
        )
        converted = list(proc.iter_frames())
    assert len(converted) == 3


def test_processor_export_gif(tmp_path):
    frames = [_make_bgr_frame() for _ in range(3)]
    fake_cap = FakeCapture(frames, fps=10.0)
    options = ConvertOptions(width=10, color=ColorMode.NONE)
    output = tmp_path / "test.gif"
    with patch("cv2.VideoCapture", return_value=fake_cap):
        proc = VideoProcessor(
            source="test.mp4",
            options=options,
            target_fps=10.0,
            temporal_smoothing=0.0,
        )
        proc.export_gif(str(output))
    assert output.exists()
    assert output.stat().st_size > 0

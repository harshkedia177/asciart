from __future__ import annotations

import numpy as np
import pytest
import time

cv2 = pytest.importorskip("cv2")

from unittest.mock import patch
from asciart.core.frame_grabber import ThreadedFrameGrabber


class FakeCapture:
    """Simulates cv2.VideoCapture for testing."""

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


def _make_frames(n: int, h: int = 20, w: int = 30) -> list[np.ndarray]:
    return [np.full((h, w, 3), i % 256, dtype=np.uint8) for i in range(n)]


def test_grabber_reads_frames():
    frames = _make_frames(5)
    fake_cap = FakeCapture(frames)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        grabber = ThreadedFrameGrabber(source="test.mp4", queue_size=3)
        grabber.start()
        time.sleep(0.2)
        frame = grabber.read()
        assert frame is not None
        assert frame.shape == (20, 30, 3)
        grabber.stop()


def test_grabber_returns_none_after_end():
    frames = _make_frames(2)
    fake_cap = FakeCapture(frames)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        grabber = ThreadedFrameGrabber(source="test.mp4", queue_size=5)
        grabber.start()
        time.sleep(0.3)
        grabbed = []
        for _ in range(10):
            f = grabber.read()
            if f is None:
                break
            grabbed.append(f)
        assert grabber.read() is None
        grabber.stop()


def test_grabber_fps_property():
    frames = _make_frames(3)
    fake_cap = FakeCapture(frames, fps=24.0)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        grabber = ThreadedFrameGrabber(source="test.mp4")
        assert grabber.fps == 24.0
        grabber.stop()


def test_grabber_frame_count():
    frames = _make_frames(10)
    fake_cap = FakeCapture(frames)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        grabber = ThreadedFrameGrabber(source="test.mp4")
        assert grabber.frame_count == 10
        grabber.stop()


def test_grabber_is_webcam():
    frames = _make_frames(3)
    fake_cap = FakeCapture(frames)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        grabber = ThreadedFrameGrabber(source=0)
        assert grabber.is_webcam is True
        grabber.stop()
    with patch("cv2.VideoCapture", return_value=FakeCapture(_make_frames(3))):
        grabber2 = ThreadedFrameGrabber(source="video.mp4")
        assert grabber2.is_webcam is False
        grabber2.stop()


def test_grabber_stop_is_idempotent():
    frames = _make_frames(3)
    fake_cap = FakeCapture(frames)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        grabber = ThreadedFrameGrabber(source="test.mp4")
        grabber.start()
        grabber.stop()
        grabber.stop()  # Should not raise

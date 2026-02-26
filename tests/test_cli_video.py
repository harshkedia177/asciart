from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from unittest.mock import patch
from typer.testing import CliRunner

from asciart.cli import app


runner = CliRunner()


class FakeCapture:
    def __init__(self, n_frames: int = 3, h: int = 60, w: int = 80, fps: float = 10.0):
        self._frames = [np.full((h, w, 3), 128, dtype=np.uint8) for _ in range(n_frames)]
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
            return 80.0
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 60.0
        return 0.0

    def release(self):
        pass


def test_video_command_exists():
    result = runner.invoke(app, ["video", "--help"])
    assert result.exit_code == 0
    assert "video" in result.output.lower() or "PATH" in result.output


def test_webcam_command_exists():
    result = runner.invoke(app, ["webcam", "--help"])
    assert result.exit_code == 0


def test_video_export_gif(tmp_path):
    fake_cap = FakeCapture(n_frames=3)
    output = tmp_path / "out.gif"
    # Create a dummy video file so Typer's exists=True check passes
    dummy_video = tmp_path / "fake_video.mp4"
    dummy_video.write_bytes(b"")
    with patch("cv2.VideoCapture", return_value=fake_cap):
        result = runner.invoke(app, [
            "video", str(dummy_video),
            "-o", str(output),
            "-w", "10",
            "--color", "none",
        ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert output.exists()
    assert output.stat().st_size > 0

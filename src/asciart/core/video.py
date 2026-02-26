from __future__ import annotations

import sys
import time
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from asciart.core.engine import convert
from asciart.core.frame_grabber import ThreadedFrameGrabber
from asciart.core.temporal import TemporalSmoother
from asciart.models import AsciiArt, ColorMode, ConvertOptions
from asciart.renderers.terminal import render_ansi
from asciart.renderers.text import render_text


class VideoProcessor:
    """Orchestrates video → ASCII conversion.

    Ties together ThreadedFrameGrabber, TemporalSmoother, engine.convert(),
    and the renderers for terminal playback and file export.

    Args:
        source: File path (str) or camera index (int).
        options: Conversion settings.
        target_fps: Desired playback frame rate.
        temporal_smoothing: EMA alpha for inter-frame smoothing (1.0 = no smoothing).
    """

    def __init__(
        self,
        source: str | int,
        options: ConvertOptions,
        target_fps: float = 30.0,
        temporal_smoothing: float = 0.3,
    ) -> None:
        self._source = source
        self._options = options
        self._target_fps = target_fps
        self._smoother = TemporalSmoother(alpha=temporal_smoothing)
        self._grabber = ThreadedFrameGrabber(source)

    @property
    def fps(self) -> float:
        return self._grabber.fps

    @property
    def frame_count(self) -> int:
        return self._grabber.frame_count

    @property
    def is_webcam(self) -> bool:
        return self._grabber.is_webcam

    def iter_frames(self) -> Iterator[AsciiArt]:
        """Yield converted AsciiArt for every frame in the source."""
        cap = cv2.VideoCapture(self._source)
        try:
            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    break
                yield self._process_frame(bgr_frame)
        finally:
            cap.release()

    def _process_frame(self, bgr_frame: np.ndarray) -> AsciiArt:
        """Convert a single BGR frame to AsciiArt via temporal smoothing + engine."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        float_frame = rgb_frame.astype(np.float64)
        smoothed = self._smoother.smooth(float_frame)
        pil_image = Image.fromarray(np.clip(smoothed, 0, 255).astype(np.uint8))
        return convert(pil_image, self._options)

    def play_terminal(self) -> None:
        """Play the video as live ASCII art in the terminal."""
        self._grabber.start()
        use_256 = self._options.color == ColorMode.ANSI256
        use_color = self._options.color != ColorMode.NONE
        frame_interval = 1.0 / self._target_fps
        frame_num = 0
        sys.stdout.write("\033[?25l")
        sys.stdout.write("\033[2J")
        try:
            while True:
                frame_start = time.monotonic()
                bgr_frame = self._grabber.read()
                if bgr_frame is None:
                    break
                art = self._process_frame(bgr_frame)
                frame_num += 1
                if use_color:
                    rendered = render_ansi(art, use_256=use_256)
                else:
                    rendered = render_text(art)
                process_time = time.monotonic() - frame_start
                actual_fps = 1.0 / process_time if process_time > 0 else 0
                total = self.frame_count
                if total > 0:
                    status = (
                        f" Frame {frame_num}/{total}"
                        f" | {actual_fps:.1f} FPS"
                        f" | {process_time * 1000:.0f}ms "
                    )
                else:
                    status = (
                        f" Frame {frame_num}"
                        f" | {actual_fps:.1f} FPS"
                        f" | {process_time * 1000:.0f}ms "
                    )
                sys.stdout.write("\033[H")
                sys.stdout.write(rendered)
                sys.stdout.write(f"\n\033[7m{status}\033[0m")
                sys.stdout.flush()
                elapsed = time.monotonic() - frame_start
                remaining = frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        finally:
            self._grabber.stop()
            sys.stdout.write("\033[?25h")
            sys.stdout.write("\033[0m\n")

    def export_gif(self, output_path: str, fps: float | None = None) -> None:
        """Export all frames as an animated GIF."""
        from asciart.renderers.video_export import export_to_gif

        target_fps = fps or self._target_fps
        frames = list(self.iter_frames())
        export_to_gif(frames, output_path, target_fps)

    def export_mp4(self, output_path: str, fps: float | None = None) -> None:
        """Export all frames as an MP4 video."""
        from asciart.renderers.video_export import export_to_mp4

        target_fps = fps or self._target_fps
        first_iter = self.iter_frames()
        first_frame = next(first_iter)

        def all_frames() -> Iterator[AsciiArt]:
            yield first_frame
            yield from first_iter

        export_to_mp4(
            all_frames(),
            output_path,
            target_fps,
            first_frame.width,
            first_frame.height,
        )

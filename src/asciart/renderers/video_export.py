from __future__ import annotations

import io
import shutil
import subprocess
from typing import Callable, Iterator

from PIL import Image

from asciart.models import AsciiArt
from asciart.renderers.image import render_png, CHAR_WIDTH, CHAR_HEIGHT


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _art_to_pil(art: AsciiArt) -> Image.Image:
    png_bytes = render_png(art)
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def export_to_gif(
    frames: list[AsciiArt],
    output_path: str,
    fps: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    if not frames:
        raise ValueError("No frames to export")

    duration_ms = int(1000.0 / fps)
    total = len(frames)
    pil_frames = []

    for i, art in enumerate(frames):
        pil_frames.append(_art_to_pil(art))
        if on_progress:
            on_progress(i + 1, total)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def export_to_mp4(
    frames: Iterator[AsciiArt],
    output_path: str,
    fps: float,
    width: int,
    height: int,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    if not check_ffmpeg():
        raise RuntimeError(
            "MP4 export requires FFmpeg. Install from https://ffmpeg.org/"
        )

    pixel_w = width * CHAR_WIDTH
    pixel_h = height * CHAR_HEIGHT

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{pixel_w}x{pixel_h}",
            "-r", str(fps),
            "-i", "pipe:",
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_num = 0
    try:
        for art in frames:
            pil_img = _art_to_pil(art)
            if pil_img.size != (pixel_w, pixel_h):
                pil_img = pil_img.resize((pixel_w, pixel_h), Image.LANCZOS)
            proc.stdin.write(pil_img.tobytes())  # type: ignore[union-attr]
            frame_num += 1
            if on_progress:
                on_progress(frame_num)
    finally:
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait()

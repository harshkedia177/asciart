from __future__ import annotations

import queue
import threading

import cv2
import numpy as np


class ThreadedFrameGrabber:
    """Reads video frames in a background thread with a bounded queue.

    Drops old frames when the consumer can't keep up.

    Args:
        source: File path (str) or camera index (int).
        queue_size: Max frames to buffer. Default 2.
    """

    def __init__(self, source: str | int, queue_size: int = 2) -> None:
        self._source = source
        self._cap = cv2.VideoCapture(source)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self._stopped = False
        self._thread: threading.Thread | None = None

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else -1

    @property
    def resolution(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def is_webcam(self) -> bool:
        return isinstance(self._source, int)

    def start(self) -> None:
        self._stopped = False
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                self._queue.put(None)
                break
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

    def read(self) -> np.ndarray | None:
        if self._stopped and self._queue.empty():
            return None
        try:
            return self._queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cap.release()

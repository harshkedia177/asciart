from __future__ import annotations

import numpy as np


class TemporalSmoother:
    """Reduces frame-to-frame flickering via exponential moving average.

    Args:
        alpha: Blend strength. 1.0 = no smoothing, 0.0 = freeze on first frame. Default 0.3.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._previous: np.ndarray | None = None

    def smooth(self, frame: np.ndarray) -> np.ndarray:
        if self._previous is None:
            self._previous = frame.copy()
            return frame

        smoothed = self._alpha * frame + (1.0 - self._alpha) * self._previous
        self._previous = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        self._previous = None

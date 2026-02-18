# src/asciart/core/edges.py
from __future__ import annotations
import numpy as np
from scipy.signal import convolve2d

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def detect_edges(gray: np.ndarray, threshold: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    gx = convolve2d(gray, SOBEL_X, mode="same", boundary="symm")
    gy = convolve2d(gray, SOBEL_Y, mode="same", boundary="symm")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angles = np.arctan2(gy, gx)
    mask = magnitude > threshold
    return mask, angles


def angle_to_char(angle: float) -> str:
    a = angle % np.pi
    if a < np.pi / 8 or a > 7 * np.pi / 8:
        return "-"
    elif a < 3 * np.pi / 8:
        return "/"
    elif a < 5 * np.pi / 8:
        return "|"
    else:
        return "\\"

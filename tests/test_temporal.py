from __future__ import annotations

import numpy as np
import numpy.testing as npt

from asciart.core.temporal import TemporalSmoother


def test_first_frame_returns_unchanged():
    smoother = TemporalSmoother(alpha=0.3)
    frame = np.array([[[100, 150, 200]]], dtype=np.float64)
    result = smoother.smooth(frame)
    npt.assert_array_equal(result, frame)


def test_smoothing_blends_with_previous():
    smoother = TemporalSmoother(alpha=0.5)
    frame1 = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    frame2 = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float64)
    smoother.smooth(frame1)
    result = smoother.smooth(frame2)
    npt.assert_allclose(result, [[[50.0, 50.0, 50.0]]])


def test_alpha_one_means_no_smoothing():
    smoother = TemporalSmoother(alpha=1.0)
    frame1 = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    frame2 = np.array([[[200.0, 200.0, 200.0]]], dtype=np.float64)
    smoother.smooth(frame1)
    result = smoother.smooth(frame2)
    npt.assert_array_equal(result, frame2)


def test_alpha_zero_freezes_on_first_frame():
    smoother = TemporalSmoother(alpha=0.0)
    frame1 = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float64)
    frame2 = np.array([[[200.0, 200.0, 200.0]]], dtype=np.float64)
    smoother.smooth(frame1)
    result = smoother.smooth(frame2)
    npt.assert_array_equal(result, frame1)


def test_reset_clears_state():
    smoother = TemporalSmoother(alpha=0.5)
    frame1 = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float64)
    frame2 = np.array([[[200.0, 200.0, 200.0]]], dtype=np.float64)
    smoother.smooth(frame1)
    smoother.reset()
    result = smoother.smooth(frame2)
    npt.assert_array_equal(result, frame2)


def test_shape_preserved():
    smoother = TemporalSmoother(alpha=0.3)
    frame = np.random.rand(40, 80, 3) * 255.0
    result = smoother.smooth(frame)
    assert result.shape == (40, 80, 3)


def test_disabled_when_alpha_one():
    smoother = TemporalSmoother(alpha=1.0)
    for _ in range(5):
        frame = np.random.rand(10, 10, 3) * 255.0
        result = smoother.smooth(frame)
        npt.assert_array_equal(result, frame)

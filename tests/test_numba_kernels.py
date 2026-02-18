import numpy as np
import pytest

# Skip all tests if numba not installed
numba = pytest.importorskip("numba")

from asciart.accel.numba_kernels import floyd_steinberg_jit, atkinson_jit


def test_floyd_steinberg_jit_matches_python():
    from asciart.core.dither import _floyd_steinberg_python
    img = np.random.randint(0, 256, (20, 40)).astype(np.float64)
    py_result = _floyd_steinberg_python(img.copy(), 4)
    jit_result = floyd_steinberg_jit(img.copy(), 4)
    np.testing.assert_allclose(py_result, jit_result, atol=1.0)


def test_atkinson_jit_matches_python():
    from asciart.core.dither import _atkinson_python
    img = np.random.randint(0, 256, (20, 40)).astype(np.float64)
    py_result = _atkinson_python(img.copy(), 4)
    jit_result = atkinson_jit(img.copy(), 4)
    np.testing.assert_allclose(py_result, jit_result, atol=1.0)


def test_jit_handles_large_image():
    img = np.random.randint(0, 256, (100, 200)).astype(np.float64)
    result = floyd_steinberg_jit(img, 10)
    assert result.shape == (100, 200)

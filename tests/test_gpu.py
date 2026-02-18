"""Tests for CuPy GPU backend integration.

All tests in this module require cupy to be installed. When cupy is not
available (the common case on dev machines), the entire module is skipped
gracefully via ``pytest.importorskip``.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from PIL import Image
from asciart.core.engine import convert
from asciart.models import ConvertOptions, ColorMode
from asciart.accel.backend import force_backend


def test_gpu_convert_basic():
    """GPU backend produces a valid AsciiArt result."""
    force_backend("cupy")
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    opts = ConvertOptions(width=20, color=ColorMode.NONE)
    result = convert(img, opts)
    assert result.width == 20


def test_gpu_results_match_cpu():
    """GPU and CPU backends produce identical character output."""
    img = Image.new("RGB", (100, 100), (100, 150, 200))
    opts = ConvertOptions(width=20, color=ColorMode.NONE)

    force_backend("numpy")
    cpu_result = convert(img, opts)

    force_backend("cupy")
    gpu_result = convert(img, opts)

    np.testing.assert_array_equal(cpu_result.char_array, gpu_result.char_array)

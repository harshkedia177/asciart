# tests/test_edges.py
import numpy as np
from asciart.core.edges import detect_edges


def test_detect_edges_returns_mask_and_angles():
    img = np.zeros((20, 20))
    img[:, 10:] = 255
    mask, angles = detect_edges(img, threshold=50)
    assert mask.shape == (20, 20)
    assert angles.shape == (20, 20)
    assert mask.dtype == bool


def test_detects_vertical_edge():
    img = np.zeros((20, 20))
    img[:, 10:] = 255
    mask, angles = detect_edges(img, threshold=30)
    assert mask[:, 9:12].any()


def test_uniform_image_no_edges():
    img = np.full((20, 20), 128.0)
    mask, angles = detect_edges(img, threshold=50)
    assert not mask.any()


def test_edge_chars_in_output():
    from PIL import Image
    from asciart.core.engine import convert
    from asciart.models import ConvertOptions, ColorMode
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        for y in range(100):
            if x < 50:
                img.putpixel((x, y), (0, 0, 0))
            else:
                img.putpixel((x, y), (255, 255, 255))
    opts = ConvertOptions(width=40, edge_detection=True, color=ColorMode.NONE)
    result = convert(img, opts)
    chars_used = {cell.char for row in result.cells for cell in row}
    edge_chars = set("|-/\\")
    assert chars_used & edge_chars

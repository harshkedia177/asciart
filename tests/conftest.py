import pytest
from PIL import Image


@pytest.fixture
def white_image():
    """A 100x100 pure white image."""
    return Image.new("RGB", (100, 100), (255, 255, 255))


@pytest.fixture
def black_image():
    """A 100x100 pure black image."""
    return Image.new("RGB", (100, 100), (0, 0, 0))


@pytest.fixture
def gradient_image():
    """A 256x1 horizontal gradient from black to white."""
    img = Image.new("RGB", (256, 1))
    for x in range(256):
        img.putpixel((x, 0), (x, x, x))
    return img


@pytest.fixture
def small_color_image():
    """A 4x4 image with distinct colors."""
    img = Image.new("RGB", (4, 4))
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (128, 128, 128), (0, 0, 0), (255, 255, 255), (128, 0, 0),
        (0, 128, 0), (0, 0, 128), (64, 64, 64), (192, 192, 192),
        (255, 128, 0), (128, 0, 255), (0, 255, 128), (64, 128, 192),
    ]
    for i, color in enumerate(colors):
        img.putpixel((i % 4, i // 4), color)
    return img

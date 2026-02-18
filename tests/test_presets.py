from asciart.presets import PRESETS, get_preset
from asciart.models import ConvertOptions, DitherMode


def test_all_presets_are_valid():
    for name, opts in PRESETS.items():
        assert isinstance(opts, ConvertOptions)
        assert isinstance(name, str)


def test_photo_preset():
    opts = get_preset("photo")
    assert opts.dither == DitherMode.FLOYD_STEINBERG
    assert opts.contrast > 1.0


def test_retro_preset():
    opts = get_preset("retro")
    assert opts.chars == " .:#"
    assert opts.dither == DitherMode.ORDERED


def test_unknown_preset_returns_none():
    assert get_preset("nonexistent") is None


def test_presets_contain_expected_names():
    expected = {"photo", "logo", "retro", "hd", "blocks", "lineart"}
    assert expected.issubset(set(PRESETS.keys()))

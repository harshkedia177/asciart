from asciart.core.ramps import RAMPS, resolve_chars


def test_resolve_known_ramp_names():
    for name, ramp in RAMPS.items():
        assert resolve_chars(name) == ramp


def test_resolve_custom_string():
    assert resolve_chars("abc123") == "abc123"
    assert resolve_chars(" @#$") == " @#$"


def test_resolve_does_not_confuse_substring():
    # A string that contains a ramp name but isn't one
    assert resolve_chars("standard123") == "standard123"


def test_all_ramps_start_with_space():
    """Every ramp should start with a space (lightest character)."""
    for name, ramp in RAMPS.items():
        assert ramp[0] == " ", f"Ramp '{name}' doesn't start with space"


def test_all_ramps_have_at_least_3_chars():
    for name, ramp in RAMPS.items():
        assert len(ramp) >= 3, f"Ramp '{name}' too short: {len(ramp)}"

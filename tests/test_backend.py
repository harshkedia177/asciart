"""Tests for the accel backend auto-detection module."""

import pytest

from asciart.accel.backend import Backend, detect_backend, force_backend, get_backend


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure every test starts with a fresh singleton."""
    force_backend("numpy")
    yield
    force_backend("numpy")



def test_detect_backend_returns_backend():
    backend = detect_backend()
    assert isinstance(backend, Backend)


def test_backend_has_numpy():
    """The xp module must expose an .array callable (numpy-compatible API)."""
    backend = detect_backend()
    assert hasattr(backend.xp, "array")


def test_get_backend_is_singleton():
    a = get_backend()
    b = get_backend()
    assert a is b


def test_backend_numpy_fallback():
    """Array operations should work on the fallback numpy backend."""
    backend = detect_backend()
    arr = backend.xp.array([1, 2, 3])
    assert arr.sum() == 6


def test_backend_has_expected_fields():
    backend = detect_backend()
    assert isinstance(backend.name, str)
    assert isinstance(backend.has_jit, bool)
    assert isinstance(backend.has_gpu, bool)


def test_force_backend_numpy():
    backend = force_backend("numpy")
    assert backend.name == "numpy"
    assert backend.has_gpu is False
    # Singleton should now reflect the forced backend
    assert get_backend() is backend


def test_force_backend_invalid_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        force_backend("nonexistent")

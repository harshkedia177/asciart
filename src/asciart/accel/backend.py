"""Auto-detect the best available compute backend: CuPy GPU -> Numba JIT -> NumPy."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

import numpy as np

_VALID_BACKENDS = ("cupy", "numba", "numpy")


@dataclass(frozen=True)
class Backend:
    """Describes the active compute backend."""

    name: str
    xp: ModuleType  # numpy-compatible array module (numpy or cupy)
    has_jit: bool
    has_gpu: bool


def _probe_cupy() -> Backend | None:
    """Try to import CuPy and verify a CUDA device is available."""
    try:
        cupy = importlib.import_module("cupy")
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        has_jit = _probe_numba() is not None
        return Backend(name="cupy", xp=cupy, has_jit=has_jit, has_gpu=True)
    except Exception:  # noqa: BLE001 — broad catch is intentional for probing
        return None


def _probe_numba() -> Backend | None:
    """Try to import Numba (CPU JIT, no GPU)."""
    try:
        importlib.import_module("numba")
        return Backend(name="numba", xp=np, has_jit=True, has_gpu=False)
    except Exception:  # noqa: BLE001
        return None


def _make_numpy_backend() -> Backend:
    """Pure NumPy fallback — always available."""
    return Backend(name="numpy", xp=np, has_jit=False, has_gpu=False)


def detect_backend() -> Backend:
    """Probe for the best available backend: CuPy GPU -> Numba JIT -> NumPy."""
    for probe in (_probe_cupy, _probe_numba):
        result = probe()
        if result is not None:
            return result
    return _make_numpy_backend()


_backend: Backend | None = None


def get_backend() -> Backend:
    """Return the cached backend singleton, detecting on first call."""
    global _backend  # noqa: PLW0603
    if _backend is None:
        _backend = detect_backend()
    return _backend


def force_backend(name: str) -> Backend:
    """Override the cached backend.

    *name* must be one of ``"cupy"``, ``"numba"``, or ``"numpy"``.
    Raises ``ValueError`` if the name is unrecognised or the backend is unavailable.
    """
    global _backend  # noqa: PLW0603

    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Must be one of {_VALID_BACKENDS}"
        )

    if name == "numpy":
        _backend = _make_numpy_backend()
    elif name == "cupy":
        result = _probe_cupy()
        if result is None:
            raise ValueError(
                f"Backend {name!r} requested but is not available on this system"
            )
        _backend = result
    elif name == "numba":
        result = _probe_numba()
        if result is None:
            raise ValueError(
                f"Backend {name!r} requested but is not available on this system"
            )
        _backend = result

    return _backend

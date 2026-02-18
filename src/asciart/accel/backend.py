"""Auto-detect the best available compute backend: CuPy GPU -> Numba JIT -> NumPy."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

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
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count < 1:
            return None
        # CuPy is usable as an array module; also check for numba JIT
        has_jit = _has_numba()
        return Backend(name="cupy", xp=cupy, has_jit=has_jit, has_gpu=True)
    except Exception:  # noqa: BLE001 — broad catch is intentional for probing
        return None


def _has_numba() -> bool:
    """Return True if numba can be imported."""
    try:
        importlib.import_module("numba")
        return True
    except Exception:  # noqa: BLE001
        return False


def _probe_numba() -> Backend | None:
    """Try to import Numba (CPU JIT, no GPU)."""
    try:
        importlib.import_module("numba")
        numpy = importlib.import_module("numpy")
        return Backend(name="numba", xp=numpy, has_jit=True, has_gpu=False)
    except Exception:  # noqa: BLE001
        return None


def _make_numpy_backend() -> Backend:
    """Pure NumPy fallback — always available."""
    numpy = importlib.import_module("numpy")
    return Backend(name="numpy", xp=numpy, has_jit=False, has_gpu=False)


def detect_backend() -> Backend:
    """Probe for the best available backend: CuPy GPU -> Numba JIT -> NumPy."""
    for probe in (_probe_cupy, _probe_numba):
        result = probe()
        if result is not None:
            return result
    return _make_numpy_backend()


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------
_backend: Backend | None = None


def get_backend() -> Backend:
    """Return the cached backend singleton, detecting on first call."""
    global _backend  # noqa: PLW0603
    if _backend is None:
        _backend = detect_backend()
    return _backend


def force_backend(name: str) -> Backend:
    """Override the cached backend.

    Parameters
    ----------
    name:
        One of ``"cupy"``, ``"numba"``, or ``"numpy"``.

    Returns
    -------
    Backend
        The newly-active backend.

    Raises
    ------
    ValueError
        If *name* is not a recognised backend.
    """
    global _backend  # noqa: PLW0603

    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Must be one of {_VALID_BACKENDS}"
        )

    builders = {
        "cupy": _probe_cupy,
        "numba": _probe_numba,
        "numpy": _make_numpy_backend,
    }

    if name == "numpy":
        _backend = _make_numpy_backend()
    else:
        result = builders[name]()
        if result is None:
            raise ValueError(
                f"Backend {name!r} requested but is not available on this system"
            )
        _backend = result

    return _backend

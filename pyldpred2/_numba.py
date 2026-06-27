"""Optional Numba acceleration shim.

The Gibbs sampler's inner sweep dominates runtime; JIT-compiling it gives a large
speed-up. If Numba is not installed we fall back to no-op decorators and run the
identical code in pure Python (just slower). All modules get their JIT decorators
from here so the with/without-Numba switch lives in one place.
"""

from __future__ import annotations

__all__ = ["HAVE_NUMBA", "_jit", "_jit_parallel", "_set_threads", "prange"]

try:
    from numba import njit as _njit, prange

    HAVE_NUMBA = True

    def _jit(func):
        return _njit(cache=True)(func)

    def _jit_parallel(func):
        return _njit(cache=True, parallel=True)(func)

    def _set_threads(ncores):
        from numba import set_num_threads
        set_num_threads(int(ncores))

except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False
    prange = range

    def _jit(func):
        return func

    def _jit_parallel(func):
        return func

    def _set_threads(ncores):
        pass

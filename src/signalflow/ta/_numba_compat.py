"""Numba compatibility layer — provides fallback when Numba is not installed.

When Numba is available, exports the real @njit decorator.
When Numba is absent, exports a no-op decorator that lets functions
run at normal Python speed.
"""

from __future__ import annotations

from typing import Any

try:
    from numba import njit as _real_njit

    NUMBA_AVAILABLE: bool = True
    njit: Any = _real_njit
except ImportError:
    NUMBA_AVAILABLE = False

    def _passthrough_njit(*args: Any, **kwargs: Any) -> Any:
        """No-op decorator that mimics numba.njit signature.

        Handles both ``@njit`` (bare) and ``@njit(cache=True)`` (factory).
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrapper(func: Any) -> Any:
            return func

        return wrapper

    njit = _passthrough_njit

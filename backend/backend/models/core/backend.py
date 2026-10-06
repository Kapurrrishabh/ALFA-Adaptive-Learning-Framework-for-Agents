"""Single import point for the array library, so a later GPU swap (cupy) is one line."""

import numpy as xp

_dtype = xp.float32


def dtype():
    return _dtype


def set_dtype(new_dtype):
    """float64 is for gradient checking only; training runs in float32 for speed."""
    global _dtype
    _dtype = xp.dtype(new_dtype)


def default_rng(seed):
    return xp.random.default_rng(seed)


_check_numerics = False


def check_numerics():
    return _check_numerics


def set_check_numerics(enabled):
    """Makes every op verify its output is finite. On in tests, off in training for speed."""
    global _check_numerics
    _check_numerics = bool(enabled)

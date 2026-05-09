"""Validation utilities for input data."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _as_numpy(x: Any, dtype: type[np.floating], copy: bool) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=dtype)
    if copy:
        arr = arr.copy()
    return arr


def validate_x(
    x: Any,
    *,
    dtype: type[np.floating] = np.float64,
    copy: bool = True,
    check_input: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.str_] | None]:
    feature_names: NDArray[np.str_] | None = None
    if pd is not None and isinstance(x, pd.DataFrame):
        feature_names = x.columns.astype(str).to_numpy()
        arr = _as_numpy(x.values, dtype, copy)
    else:
        arr = _as_numpy(x, dtype, copy)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape={arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("X must be non-empty with at least 1 sample and 1 feature.")
    if check_input and not np.isfinite(arr).all():
        raise ValueError("X contains NaN or infinite values.")
    return arr, feature_names


def validate_y(
    y: Any,
    *,
    dtype: type[np.floating] = np.float64,
    copy: bool = True,
    check_input: bool = True,
) -> NDArray[np.float64]:
    if pd is not None and isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y DataFrame must have exactly one column.")
        arr = _as_numpy(y.iloc[:, 0].values, dtype, copy)
    elif pd is not None and isinstance(y, pd.Series):
        arr = _as_numpy(y.values, dtype, copy)
    else:
        arr = _as_numpy(y, dtype, copy)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"y must be 1D or single-column. Got shape={arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("y must be non-empty.")
    if check_input and not np.isfinite(arr).all():
        raise ValueError("y contains NaN or infinite values.")
    return arr


def validate_xy_lengths(x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y row mismatch: X has {x.shape[0]} rows but y has {y.shape[0]}."
        )


def validate_degree(degree: int) -> None:
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}.")

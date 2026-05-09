"""Small metric helpers with sklearn-free fallback."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_squared_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(diff * diff))


def r2_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

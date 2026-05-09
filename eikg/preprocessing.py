"""Preprocessing helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class StandardScalerLite:
    """Lightweight scaler with sklearn-like behavior."""

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64]) -> StandardScalerLite:
        self.mean_ = np.mean(x, axis=0) if self.with_mean else np.zeros(x.shape[1], dtype=x.dtype)
        if self.with_std:
            scale = np.std(x, axis=0)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(x.shape[1], dtype=x.dtype)
        return self

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        return x * self.scale_ + self.mean_


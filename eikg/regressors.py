"""Regressors for compact Kolmogorov-Gabor elementary images."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import mean_squared_error, r2_score
from .preprocessing import StandardScalerLite
from .validation import validate_degree, validate_x, validate_xy_lengths, validate_y

try:
    from scipy.linalg import lstsq as scipy_lstsq
except Exception:  # pragma: no cover - optional dependency
    scipy_lstsq = None

try:
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    from sklearn.model_selection import KFold
except Exception:  # pragma: no cover - optional dependency
    class BaseEstimator:  # type: ignore[no-redef]
        """Fallback BaseEstimator when sklearn is unavailable."""

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return self.__dict__.copy()

        def set_params(self, **params: Any) -> BaseEstimator:
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class RegressorMixin:  # type: ignore[no-redef]
        """Fallback mixin when sklearn is unavailable."""

    clone = None
    KFold = None


class EIKGPolynomialRegressor(BaseEstimator, RegressorMixin):
    """Two-stage EIKG regressor with numerically stable least squares.

    The model is trained in two stages:

    1. Linear latent model:
       ``z = b0 + b1*x1 + ... + bm*xm``
    2. Polynomial map from latent prediction:
       ``y_hat = a0 + a1*z + a2*z^2 + ... + ad*z^d``

    Parameters
    ----------
    degree : int, default=2
        Degree of latent polynomial expansion.
    regularization : {"none", "ridge", None}, default="none"
        Solver regularization mode.
    alpha_ridge : float, default=1e-8
        Ridge regularization strength for ``regularization="ridge"``.
    fit_intercept : bool, default=True
        Whether to fit intercept terms in both stages.
    scale : bool, default=True
        Standardize X before first-stage regression.
    scale_y : bool, default=False
        Standardize y during training and invert at prediction.
    normalize_latent : bool, default=True
        Normalize latent predictions before building powers.
    dtype : numpy floating type, default=np.float64
        Numeric dtype used internally.
    copy : bool, default=True
        Whether to copy input arrays during validation.
    check_input : bool, default=True
        Whether to validate finite values and shape constraints.
    lstsq_rcond : float or None, default=None
        rcond passed to lstsq backend.
    """

    def __init__(
        self,
        degree: int = 2,
        regularization: str | None = "none",
        alpha_ridge: float = 1e-8,
        fit_intercept: bool = True,
        scale: bool = True,
        scale_y: bool = False,
        normalize_latent: bool = True,
        dtype: type[np.floating] = np.float64,
        copy: bool = True,
        check_input: bool = True,
        lstsq_rcond: float | None = None,
    ) -> None:
        self.degree = degree
        self.regularization = regularization
        self.alpha_ridge = alpha_ridge
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.scale_y = scale_y
        self.normalize_latent = normalize_latent
        self.dtype = dtype
        self.copy = copy
        self.check_input = check_input
        self.lstsq_rcond = lstsq_rcond

    def fit(self, x: Any, y: Any) -> EIKGPolynomialRegressor:
        validate_degree(self.degree)
        x_arr, feature_names = validate_x(
            x, dtype=self.dtype, copy=self.copy, check_input=self.check_input
        )
        y_arr = validate_y(y, dtype=self.dtype, copy=self.copy, check_input=self.check_input)
        validate_xy_lengths(x_arr, y_arr)
        self.n_features_in_ = x_arr.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        self.x_scaler_: StandardScalerLite | None
        if self.scale:
            self.x_scaler_ = StandardScalerLite().fit(x_arr)
            x_model = self.x_scaler_.transform(x_arr)
        else:
            self.x_scaler_ = None
            x_model = x_arr

        self.y_scaler_: StandardScalerLite | None
        if self.scale_y:
            self.y_scaler_ = StandardScalerLite().fit(y_arr.reshape(-1, 1))
            y_model = self.y_scaler_.transform(y_arr.reshape(-1, 1)).ravel()
        else:
            self.y_scaler_ = None
            y_model = y_arr

        beta, intercept1, rank1, sing1, cond1 = self._solve_least_squares(x_model, y_model)
        self.beta_ = beta
        self.intercept_1_ = intercept1
        self.rank_ = rank1
        self.singular_values_ = sing1
        self.condition_number_ = cond1

        z_train = x_model @ self.beta_ + self.intercept_1_
        phi = self._build_latent_features(z_train, fit=True)
        alpha, intercept2, rank2, sing2, _ = self._solve_least_squares(phi, y_model)
        self.alpha_ = alpha
        self.intercept_2_ = intercept2
        self.rank_latent_ = rank2
        self.singular_values_latent_ = sing2
        self.degree_ = int(self.degree)
        self.is_fitted_ = True
        return self

    def predict(self, x: Any) -> NDArray[np.float64]:
        self._check_is_fitted()
        x_arr, feature_names = validate_x(
            x, dtype=self.dtype, copy=self.copy, check_input=self.check_input
        )
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x_arr.shape[1]} features, expected {self.n_features_in_}."
            )
        if hasattr(self, "feature_names_in_") and feature_names is not None:
            if not np.array_equal(feature_names, self.feature_names_in_):
                raise ValueError("X columns at predict must match training feature names/order.")
        x_model = self.x_scaler_.transform(x_arr) if self.x_scaler_ is not None else x_arr
        z = x_model @ self.beta_ + self.intercept_1_
        phi = self._build_latent_features(z, fit=False)
        y_pred = phi @ self.alpha_ + self.intercept_2_
        if self.y_scaler_ is not None:
            y_pred = self.y_scaler_.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred

    def score(self, x: Any, y: Any) -> float:
        y_true = validate_y(y, dtype=self.dtype, copy=False, check_input=self.check_input)
        y_pred = self.predict(x)
        return r2_score(y_true, y_pred)

    def _build_latent_features(
        self, y_hat: NDArray[np.float64], *, fit: bool = False
    ) -> NDArray[np.float64]:
        z = np.asarray(y_hat, dtype=self.dtype)
        if fit and self.normalize_latent:
            self.latent_mean_ = float(np.mean(z))
            self.latent_scale_ = float(np.std(z))
            if self.latent_scale_ == 0.0:
                self.latent_scale_ = 1.0
        if self.normalize_latent:
            if not hasattr(self, "latent_mean_") or not hasattr(self, "latent_scale_"):
                raise RuntimeError("Latent normalization parameters are missing.")
            z = (z - self.latent_mean_) / self.latent_scale_
        phi = np.empty((z.shape[0], self.degree), dtype=self.dtype)
        cur = z.copy()
        for degree_idx in range(self.degree):
            if degree_idx > 0:
                cur = cur * z
            if self.check_input and not np.isfinite(cur).all():
                raise FloatingPointError(
                    "Overflow/invalid values while building latent polynomial features. "
                    "Try lower degree, enable latent normalization, or use stronger regularization."
                )
            phi[:, degree_idx] = cur
        return phi

    def _solve_least_squares(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, int, NDArray[np.float64], float]:
        regularization = "none" if self.regularization is None else str(self.regularization).lower()
        if regularization not in {"none", "ridge"}:
            raise ValueError(f"regularization must be one of None/'none'/'ridge', got {self.regularization}.")
        if regularization == "ridge" and self.alpha_ridge < 0:
            raise ValueError("alpha_ridge must be >= 0.")

        x_in = np.asarray(x, dtype=self.dtype)
        y_in = np.asarray(y, dtype=self.dtype)
        if self.fit_intercept:
            x_mean = np.mean(x_in, axis=0)
            y_mean = float(np.mean(y_in))
            x_centered = x_in - x_mean
            y_centered = y_in - y_mean
        else:
            x_mean = np.zeros(x_in.shape[1], dtype=self.dtype)
            y_mean = 0.0
            x_centered = x_in
            y_centered = y_in

        if regularization == "ridge":
            gram = x_centered.T @ x_centered
            rhs = x_centered.T @ y_centered
            eye = np.eye(gram.shape[0], dtype=self.dtype)
            coef = np.linalg.solve(gram + self.alpha_ridge * eye, rhs)
            rank = int(np.linalg.matrix_rank(x_centered))
            singular_values = np.linalg.svd(x_centered, compute_uv=False)
        else:
            if scipy_lstsq is not None:
                coef, _, rank, singular_values = scipy_lstsq(
                    x_centered, y_centered, cond=self.lstsq_rcond
                )
            else:
                coef, _, rank, singular_values = np.linalg.lstsq(
                    x_centered, y_centered, rcond=self.lstsq_rcond
                )
            coef = np.asarray(coef, dtype=self.dtype)
            singular_values = np.asarray(singular_values, dtype=self.dtype)

        intercept = y_mean - float(x_mean @ coef) if self.fit_intercept else 0.0
        if singular_values.size > 0 and singular_values[-1] > 0:
            cond = float(singular_values[0] / singular_values[-1])
        else:
            cond = float(np.inf)
        return coef, intercept, int(rank), singular_values, cond

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Estimator is not fitted. Call fit(X, y) first.")


class EIKGPolynomialRegressorCV(BaseEstimator, RegressorMixin):
    """Degree selection wrapper for EIKGPolynomialRegressor."""

    def __init__(
        self,
        max_degree: int = 6,
        scoring: str = "neg_mean_squared_error",
        cv: int = 5,
        **regressor_kwargs: Any,
    ) -> None:
        self.max_degree = max_degree
        self.scoring = scoring
        self.cv = cv
        self.regressor_kwargs = regressor_kwargs

    def fit(self, x: Any, y: Any) -> EIKGPolynomialRegressorCV:
        if self.max_degree < 1:
            raise ValueError(f"max_degree must be >= 1, got {self.max_degree}.")
        if self.scoring not in {"neg_mean_squared_error", "r2"}:
            raise ValueError("scoring must be one of {'neg_mean_squared_error', 'r2'}.")
        x_arr, _ = validate_x(x, dtype=np.float64, copy=True, check_input=True)
        y_arr = validate_y(y, dtype=np.float64, copy=True, check_input=True)
        validate_xy_lengths(x_arr, y_arr)
        if self.cv < 2:
            raise ValueError("cv must be >= 2.")

        scores: list[float] = []
        for degree in range(1, self.max_degree + 1):
            fold_scores = self._cv_score_degree(x_arr, y_arr, degree)
            scores.append(float(np.mean(fold_scores)))
        best_idx = int(np.argmax(np.asarray(scores)))
        self.selected_degree_ = best_idx + 1
        self.cv_scores_ = scores
        self.best_score_ = scores[best_idx]

        self.estimator_ = EIKGPolynomialRegressor(
            degree=self.selected_degree_, **self.regressor_kwargs
        ).fit(x, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        self.is_fitted_ = True
        return self

    def predict(self, x: Any) -> NDArray[np.float64]:
        self._check_is_fitted()
        return self.estimator_.predict(x)

    def score(self, x: Any, y: Any) -> float:
        self._check_is_fitted()
        return self.estimator_.score(x, y)

    def _cv_score_degree(
        self, x: NDArray[np.float64], y: NDArray[np.float64], degree: int
    ) -> list[float]:
        model = EIKGPolynomialRegressor(degree=degree, **self.regressor_kwargs)
        n_samples = x.shape[0]
        if KFold is not None:
            splitter = KFold(n_splits=self.cv, shuffle=False)
            indices = splitter.split(x, y)
        else:
            fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
            fold_sizes[: n_samples % self.cv] += 1
            starts = np.cumsum(np.concatenate(([0], fold_sizes[:-1])))
            indices = []
            for start, fold_size in zip(starts, fold_sizes):
                test_idx = np.arange(start, start + fold_size)
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[test_idx] = False
                train_idx = np.arange(n_samples)[train_mask]
                indices.append((train_idx, test_idx))

        fold_scores: list[float] = []
        for train_idx, test_idx in indices:
            cur_model = clone(model) if clone is not None else EIKGPolynomialRegressor(
                degree=degree, **self.regressor_kwargs
            )
            cur_model.fit(x[train_idx], y[train_idx])
            pred = cur_model.predict(x[test_idx])
            if self.scoring == "r2":
                fold_scores.append(r2_score(y[test_idx], pred))
            else:
                fold_scores.append(-mean_squared_error(y[test_idx], pred))
        return fold_scores

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Estimator is not fitted. Call fit(X, y) first.")

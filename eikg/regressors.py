"""Regressors for compact Kolmogorov-Gabor elementary images."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import mean_squared_error, r2_score
from .preprocessing import StandardScalerLite
from .validation import (
    validate_degree,
    validate_floating_dtype,
    validate_positive_integer,
    validate_x,
    validate_xy_lengths,
    validate_y,
)

try:
    from scipy.linalg import lstsq as scipy_lstsq
except Exception:  # pragma: no cover - optional dependency
    scipy_lstsq = None

try:
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    from sklearn.exceptions import NotFittedError
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

    class NotFittedError(RuntimeError):  # type: ignore[no-redef]
        """Fallback fitted-state error when sklearn is unavailable."""

    clone = None
    KFold = None


def _checked_affine_prediction(
    x: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    intercept: float,
    *,
    context: str,
) -> NDArray[np.float64]:
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            prediction = x @ coefficients + intercept
    except FloatingPointError as exc:
        raise FloatingPointError(f"Overflow or invalid values while {context}.") from exc
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"Non-finite values produced while {context}.")
    return np.asarray(prediction)


def _stable_finite_mean(values: list[float]) -> float:
    """Average finite scores without overflowing their intermediate sum."""

    array = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        result = scale * float(np.mean(array / scale))
    if not np.isfinite(result):
        raise FloatingPointError("Cross-validation produced a non-finite mean score.")
    return result


def _validate_boolean_parameters(**parameters: Any) -> None:
    """Reject truthy/falsy stand-ins for public boolean parameters."""

    for name, value in parameters.items():
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean, got {value!r}.")


class EIKGPolynomialRegressor(RegressorMixin, BaseEstimator):
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
        self._clear_fitted_state()
        try:
            return self._fit(x, y)
        except Exception:
            self._clear_fitted_state()
            raise

    def _fit(self, x: Any, y: Any) -> EIKGPolynomialRegressor:
        validate_degree(self.degree)
        validate_floating_dtype(self.dtype)
        _validate_boolean_parameters(
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            copy=self.copy,
            check_input=self.check_input,
        )
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

        z_train = _checked_affine_prediction(
            x_model,
            self.beta_,
            self.intercept_1_,
            context="computing the training latent projection",
        )
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
            raise ValueError(f"X has {x_arr.shape[1]} features, expected {self.n_features_in_}.")
        if hasattr(self, "feature_names_in_") and feature_names is not None:
            if not np.array_equal(feature_names, self.feature_names_in_):
                raise ValueError("X columns at predict must match training feature names/order.")
        x_model = self.x_scaler_.transform(x_arr) if self.x_scaler_ is not None else x_arr
        z = _checked_affine_prediction(
            x_model,
            self.beta_,
            self.intercept_1_,
            context="computing the latent projection",
        )
        phi = self._build_latent_features(z, fit=False)
        y_pred = _checked_affine_prediction(
            phi,
            self.alpha_,
            self.intercept_2_,
            context="computing the polynomial prediction",
        )
        if self.y_scaler_ is not None:
            y_pred = self.y_scaler_.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred

    def score(self, x: Any, y: Any) -> float:
        y_true = validate_y(y, dtype=self.dtype, copy=False, check_input=self.check_input)
        y_pred = self.predict(x)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"X and y row mismatch: X has {y_pred.shape[0]} rows but y has {y_true.shape[0]}."
            )
        return r2_score(y_true, y_pred)

    def _build_latent_features(
        self, y_hat: NDArray[np.float64], *, fit: bool = False
    ) -> NDArray[np.float64]:
        z = np.asarray(y_hat, dtype=self.dtype)
        if fit and self.normalize_latent:
            self.latent_scaler_ = StandardScalerLite().fit(z.reshape(-1, 1))
            assert self.latent_scaler_.mean_ is not None
            assert self.latent_scaler_.scale_ is not None
            self.latent_mean_ = float(self.latent_scaler_.mean_[0])
            self.latent_scale_ = float(self.latent_scaler_.scale_[0])
        if self.normalize_latent:
            if not hasattr(self, "latent_scaler_"):
                raise RuntimeError("Latent normalization parameters are missing.")
            z = self.latent_scaler_.transform(z.reshape(-1, 1)).ravel()
        phi = np.empty((z.shape[0], self.degree), dtype=self.dtype)
        cur = z.copy()
        for degree_idx in range(self.degree):
            if degree_idx > 0:
                try:
                    with np.errstate(over="raise", invalid="raise", under="ignore"):
                        cur = cur * z
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        "Overflow/invalid values while building latent polynomial features. "
                        "Try lower degree, enable latent normalization, or use stronger "
                        "regularization."
                    ) from exc
            if not np.isfinite(cur).all():
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
            raise ValueError(
                f"regularization must be one of None/'none'/'ridge', got {self.regularization}."
            )
        ridge_alpha = 0.0
        if regularization == "ridge":
            if isinstance(self.alpha_ridge, (bool, np.bool_)) or not isinstance(
                self.alpha_ridge, Real
            ):
                raise ValueError("alpha_ridge must be a finite number >= 0.")
            ridge_alpha = float(self.alpha_ridge)
            if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
                raise ValueError("alpha_ridge must be a finite number >= 0.")
        if self.lstsq_rcond is not None:
            if isinstance(self.lstsq_rcond, (bool, np.bool_)) or not isinstance(
                self.lstsq_rcond, Real
            ):
                raise ValueError("lstsq_rcond must be None or a finite number >= 0.")
            if not np.isfinite(self.lstsq_rcond) or self.lstsq_rcond < 0:
                raise ValueError("lstsq_rcond must be None or a finite number >= 0.")

        x_in = np.asarray(x, dtype=self.dtype)
        y_in = np.asarray(y, dtype=self.dtype)
        if self.fit_intercept:
            x_centerer = StandardScalerLite(with_std=False).fit(x_in)
            y_centerer = StandardScalerLite(with_std=False).fit(y_in.reshape(-1, 1))
            assert x_centerer.mean_ is not None
            assert y_centerer.mean_ is not None
            x_mean = x_centerer.mean_
            y_mean = float(y_centerer.mean_[0])
            x_centered = x_centerer.transform(x_in)
            y_centered = y_centerer.transform(y_in.reshape(-1, 1)).ravel()
        else:
            x_mean = np.zeros(x_in.shape[1], dtype=self.dtype)
            y_mean = 0.0
            x_centered = x_in
            y_centered = y_in

        if regularization == "ridge":
            # Apply the Ridge filter to one thin SVD. This avoids normal equations,
            # a dense p-by-p penalty matrix, and duplicate decompositions.
            u, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
            singular_values = np.asarray(singular_values, dtype=self.dtype)
            if singular_values.size:
                relative_cutoff = (
                    float(self.lstsq_rcond)
                    if self.lstsq_rcond is not None
                    else np.finfo(self.dtype).eps * max(x_centered.shape)
                )
                cutoff = relative_cutoff * float(singular_values[0])
            else:
                cutoff = 0.0
            retained = singular_values > cutoff
            rank = int(np.count_nonzero(retained))
            factors = np.zeros_like(singular_values)
            if ridge_alpha > 0.0:
                positive = singular_values > 0.0
                with np.errstate(over="ignore", divide="ignore", invalid="raise"):
                    denominator = (
                        singular_values[positive] + ridge_alpha / singular_values[positive]
                    )
                    factors[positive] = 1.0 / denominator
            else:
                factors[retained] = 1.0 / singular_values[retained]
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                    coef = vt.T @ (factors * (u.T @ y_centered))
            except FloatingPointError as exc:
                raise FloatingPointError(
                    "Ridge solve produced overflow or invalid coefficient values."
                ) from exc
            coef = np.asarray(coef, dtype=self.dtype)
        else:
            if scipy_lstsq is not None:
                coef, _, rank_result, singular_values = scipy_lstsq(
                    x_centered, y_centered, cond=self.lstsq_rcond
                )
            else:
                coef, _, rank_result, singular_values = np.linalg.lstsq(
                    x_centered, y_centered, rcond=self.lstsq_rcond
                )
            rank = int(rank_result)
            coef = np.asarray(coef, dtype=self.dtype)
            singular_values = np.asarray(singular_values, dtype=self.dtype)

        try:
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                intercept = y_mean - float(x_mean @ coef) if self.fit_intercept else 0.0
        except FloatingPointError as exc:
            raise FloatingPointError(
                "The fitted intercept overflowed; scale X or y before fitting."
            ) from exc
        if not np.isfinite(coef).all() or not np.isfinite(intercept):
            raise FloatingPointError("Least-squares fitting produced non-finite coefficients.")
        if singular_values.size > 0 and singular_values[-1] > 0:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                cond = float(singular_values[0] / singular_values[-1])
        else:
            cond = float(np.inf)
        return coef, intercept, int(rank), singular_values, cond

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError("Estimator is not fitted. Call fit(X, y) first.")

    def _clear_fitted_state(self) -> None:
        fitted_attributes = (
            "n_features_in_",
            "feature_names_in_",
            "x_scaler_",
            "y_scaler_",
            "beta_",
            "intercept_1_",
            "rank_",
            "singular_values_",
            "condition_number_",
            "latent_mean_",
            "latent_scale_",
            "latent_scaler_",
            "alpha_",
            "intercept_2_",
            "rank_latent_",
            "singular_values_latent_",
            "degree_",
            "is_fitted_",
        )
        for attribute in fitted_attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)


class EIKGPolynomialRegressorCV(RegressorMixin, BaseEstimator):
    """Degree selection wrapper for :class:`EIKGPolynomialRegressor`.

    In addition to mean cross-validation scores, the fitted estimator retains
    per-fold scores and out-of-fold predictions for the selected degree. Every
    row in ``oof_predictions_`` is predicted by a fold model that was not fitted
    on that row. Because the degree is selected using all fold scores, these are
    post-selection OOF predictions rather than an unbiased estimate of the full
    tuning procedure.
    """

    def __init__(
        self,
        max_degree: int = 6,
        scoring: str = "neg_mean_squared_error",
        cv: int = 5,
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
        self.max_degree = max_degree
        self.scoring = scoring
        self.cv = cv
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

    def fit(self, x: Any, y: Any) -> EIKGPolynomialRegressorCV:
        self._clear_fitted_state()
        max_degree = validate_positive_integer(self.max_degree, name="max_degree")
        if self.scoring not in {"neg_mean_squared_error", "r2"}:
            raise ValueError("scoring must be one of {'neg_mean_squared_error', 'r2'}.")
        validate_floating_dtype(self.dtype)
        _validate_boolean_parameters(
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            copy=self.copy,
            check_input=self.check_input,
        )
        x_arr, _ = validate_x(x, dtype=self.dtype, copy=self.copy, check_input=self.check_input)
        y_arr = validate_y(y, dtype=self.dtype, copy=self.copy, check_input=self.check_input)
        validate_xy_lengths(x_arr, y_arr)
        cv = validate_positive_integer(self.cv, name="cv", minimum=2)
        if cv > x_arr.shape[0]:
            raise ValueError(f"cv={cv} cannot exceed the number of samples ({x_arr.shape[0]}).")
        if self.scoring == "r2" and x_arr.shape[0] // cv < 2:
            raise ValueError("scoring='r2' requires at least 2 validation samples in every fold.")

        splits = self._make_splits(x_arr.shape[0], cv=cv)
        scores: list[float] = []
        all_fold_scores: list[list[float]] = []
        best_oof_predictions: NDArray[np.float64] | None = None
        best_idx = 0
        for degree in range(1, max_degree + 1):
            fold_scores, oof_predictions = self._cross_validate_degree(
                x_arr,
                y_arr,
                degree,
                splits=splits,
            )
            mean_score = _stable_finite_mean(fold_scores)
            scores.append(mean_score)
            all_fold_scores.append(fold_scores)
            if best_oof_predictions is None or mean_score > scores[best_idx]:
                best_idx = degree - 1
                best_oof_predictions = oof_predictions.copy()
        if best_oof_predictions is None:  # pragma: no cover - max_degree is positive
            raise RuntimeError("Cross-validation did not evaluate any polynomial degree.")
        selected_degree = best_idx + 1
        estimator = self._make_estimator(selected_degree).fit(x, y)
        self.selected_degree_ = selected_degree
        self.cv_scores_ = scores
        self.cv_fold_scores_ = all_fold_scores
        self.best_score_ = scores[best_idx]
        self.oof_predictions_ = best_oof_predictions
        self.estimator_ = estimator
        self.n_features_in_ = estimator.n_features_in_
        if hasattr(estimator, "feature_names_in_"):
            self.feature_names_in_ = estimator.feature_names_in_.copy()
        self.is_fitted_ = True
        return self

    def predict(self, x: Any) -> NDArray[np.float64]:
        self._check_is_fitted()
        return self.estimator_.predict(x)

    def score(self, x: Any, y: Any) -> float:
        self._check_is_fitted()
        return self.estimator_.score(x, y)

    def _cross_validate_degree(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        degree: int,
        *,
        splits: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    ) -> tuple[list[float], NDArray[np.float64]]:
        model = self._make_estimator(degree)
        fold_scores: list[float] = []
        oof_predictions = np.empty(x.shape[0], dtype=self.dtype)
        assigned = np.zeros(x.shape[0], dtype=bool)
        for train_idx, test_idx in splits:
            if np.any(assigned[test_idx]):  # pragma: no cover - guarded by split construction
                raise RuntimeError("Cross-validation assigned a training row more than once.")
            cur_model = clone(model) if clone is not None else self._make_estimator(degree)
            cur_model.fit(x[train_idx], y[train_idx])
            pred = np.asarray(cur_model.predict(x[test_idx]), dtype=self.dtype)
            if not np.isfinite(pred).all():
                raise FloatingPointError(
                    f"Degree {degree} produced non-finite out-of-fold predictions."
                )
            oof_predictions[test_idx] = pred
            assigned[test_idx] = True
            if self.scoring == "r2":
                score = r2_score(y[test_idx], pred)
            else:
                score = -mean_squared_error(y[test_idx], pred)
            if not np.isfinite(score):
                raise FloatingPointError(
                    f"Degree {degree} produced a non-finite cross-validation score."
                )
            fold_scores.append(float(score))
        if not np.all(assigned):  # pragma: no cover - guarded by split construction
            raise RuntimeError("Cross-validation did not predict every training row exactly once.")
        return fold_scores, oof_predictions

    def _make_splits(
        self, n_samples: int, *, cv: int
    ) -> list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        if KFold is not None:
            splitter = KFold(n_splits=cv, shuffle=False)
            return [
                (np.asarray(train_idx), np.asarray(test_idx))
                for train_idx, test_idx in splitter.split(np.arange(n_samples))
            ]

        fold_sizes = np.full(cv, n_samples // cv, dtype=int)
        fold_sizes[: n_samples % cv] += 1
        starts = np.cumsum(np.concatenate(([0], fold_sizes[:-1])))
        splits: list[tuple[NDArray[np.int_], NDArray[np.int_]]] = []
        for start, fold_size in zip(starts, fold_sizes):
            test_idx = np.arange(start, start + fold_size)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.arange(n_samples)[train_mask]
            splits.append((train_idx, test_idx))
        return splits

    def _make_estimator(self, degree: int) -> EIKGPolynomialRegressor:
        return EIKGPolynomialRegressor(
            degree=degree,
            regularization=self.regularization,
            alpha_ridge=self.alpha_ridge,
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            dtype=self.dtype,
            copy=self.copy,
            check_input=self.check_input,
            lstsq_rcond=self.lstsq_rcond,
        )

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError("Estimator is not fitted. Call fit(X, y) first.")

    def _clear_fitted_state(self) -> None:
        fitted_attributes = (
            "selected_degree_",
            "cv_scores_",
            "cv_fold_scores_",
            "best_score_",
            "oof_predictions_",
            "estimator_",
            "n_features_in_",
            "feature_names_in_",
            "is_fitted_",
        )
        for attribute in fitted_attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)

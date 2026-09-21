import numpy as np
import pytest

from eikg.metrics import mean_squared_error
from eikg.regressors import (
    EIKGPolynomialRegressor,
    EIKGPolynomialRegressorCV,
    NotFittedError,
)


def make_data(n_samples: int = 47, seed: int = 901):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, 3))
    latent = 0.4 + 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.15 * x[:, 2]
    y = latent + 0.1 * latent**2 + rng.normal(scale=0.03, size=n_samples)
    return x, y


def contiguous_splits(n_samples: int, cv: int):
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[: n_samples % cv] += 1
    indices = np.arange(n_samples)
    current = 0
    splits = []
    for fold_size in fold_sizes:
        validation = indices[current : current + fold_size]
        training = np.concatenate((indices[:current], indices[current + fold_size :]))
        splits.append((training, validation))
        current += int(fold_size)
    return splits


def test_cv_exposes_manual_fold_scores_and_selected_degree_oof() -> None:
    x, y = make_data()
    cv = 3
    max_degree = 3
    parameters = {
        "regularization": "ridge",
        "alpha_ridge": 1e-5,
        "scale": True,
    }
    model = EIKGPolynomialRegressorCV(
        max_degree=max_degree,
        cv=cv,
        scoring="neg_mean_squared_error",
        **parameters,
    ).fit(x, y)

    expected_scores: list[list[float]] = []
    expected_oof_by_degree: list[np.ndarray] = []
    for degree in range(1, max_degree + 1):
        fold_scores: list[float] = []
        oof = np.empty(x.shape[0])
        for training, validation in contiguous_splits(x.shape[0], cv):
            fold_model = EIKGPolynomialRegressor(degree=degree, **parameters).fit(
                x[training], y[training]
            )
            prediction = fold_model.predict(x[validation])
            oof[validation] = prediction
            fold_scores.append(-mean_squared_error(y[validation], prediction))
        expected_scores.append(fold_scores)
        expected_oof_by_degree.append(oof)

    expected_means = [float(np.mean(scores)) for scores in expected_scores]
    selected_index = int(np.argmax(expected_means))
    np.testing.assert_allclose(model.cv_fold_scores_, expected_scores)
    np.testing.assert_allclose(model.cv_scores_, expected_means)
    np.testing.assert_allclose(model.oof_predictions_, expected_oof_by_degree[selected_index])
    assert model.selected_degree_ == selected_index + 1
    assert model.best_score_ == pytest.approx(expected_means[selected_index])
    assert model.oof_predictions_.shape == (x.shape[0],)
    assert np.isfinite(model.oof_predictions_).all()


def test_cv_refit_replaces_oof_and_failed_refit_clears_diagnostics() -> None:
    x_first, y_first = make_data(n_samples=40, seed=902)
    x_second, y_second = make_data(n_samples=29, seed=903)
    model = EIKGPolynomialRegressorCV(max_degree=2, cv=2).fit(x_first, y_first)

    model.fit(x_second, y_second)
    assert model.oof_predictions_.shape == (29,)
    assert len(model.cv_fold_scores_) == 2

    model.cv = 30
    with pytest.raises(ValueError, match="cannot exceed"):
        model.fit(x_second, y_second)

    assert not hasattr(model, "oof_predictions_")
    assert not hasattr(model, "cv_fold_scores_")
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(x_second)

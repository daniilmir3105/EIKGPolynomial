import numpy as np
import pytest

from eikg.regressors import (
    EIKGPolynomialRegressor,
    EIKGPolynomialRegressorCV,
    NotFittedError,
)


def make_data(n: int = 120, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    linear = 1.0 + 2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.4 * x[:, 2]
    y = linear + 0.08 * linear**2 + rng.normal(scale=0.1, size=n)
    return x, y


def test_fit_predict_ndarray() -> None:
    x, y = make_data()
    model = EIKGPolynomialRegressor(degree=2)
    model.fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (x.shape[0],)
    assert np.isfinite(pred).all()


def test_fit_predict_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    x, y = make_data()
    x_df = pd.DataFrame(x, columns=["f1", "f2", "f3"])
    y_s = pd.Series(y)
    model = EIKGPolynomialRegressor(degree=2)
    model.fit(x_df, y_s)
    pred = model.predict(x_df)
    assert pred.shape == (len(x_df),)
    assert hasattr(model, "feature_names_in_")


def test_predict_before_fit_raises() -> None:
    x, _ = make_data()
    model = EIKGPolynomialRegressor()
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(x)


def test_degree_less_than_one_raises() -> None:
    x, y = make_data()
    model = EIKGPolynomialRegressor(degree=0)
    with pytest.raises(ValueError, match="degree must be >="):
        model.fit(x, y)


def test_nan_raises() -> None:
    x, y = make_data()
    x[0, 0] = np.nan
    model = EIKGPolynomialRegressor()
    with pytest.raises(ValueError, match="contains NaN"):
        model.fit(x, y)


def test_complex_data_raises() -> None:
    x, y = make_data()
    with pytest.raises(ValueError, match="Complex"):
        EIKGPolynomialRegressor().fit(x.astype(np.complex128) + 1j, y)


def test_multicollinearity_not_crash() -> None:
    rng = np.random.default_rng(10)
    x1 = rng.normal(size=80)
    x2 = 2.0 * x1 + 1e-10 * rng.normal(size=80)
    x3 = rng.normal(size=80)
    x = np.c_[x1, x2, x3]
    y = 0.5 + 1.1 * x1 - 0.3 * x3 + rng.normal(scale=0.01, size=80)
    model = EIKGPolynomialRegressor(degree=2, regularization="ridge", alpha_ridge=1e-5)
    model.fit(x, y)
    pred = model.predict(x)
    assert np.isfinite(pred).all()


def test_ridge_path_runs() -> None:
    x, y = make_data()
    model = EIKGPolynomialRegressor(regularization="ridge", alpha_ridge=1e-4)
    model.fit(x, y)
    assert hasattr(model, "beta_")


def test_ridge_with_zero_alpha_handles_rank_deficiency() -> None:
    x, y = make_data()
    x = np.column_stack((x, x[:, 0], 2.0 * x[:, 0]))
    model = EIKGPolynomialRegressor(regularization="ridge", alpha_ridge=0.0)

    prediction = model.fit(x, y).predict(x)

    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("alpha_ridge", [-1.0, np.inf, np.nan, True, "0.1"])
def test_ridge_rejects_invalid_alpha(alpha_ridge: object) -> None:
    x, y = make_data()

    with pytest.raises(ValueError, match="alpha_ridge"):
        EIKGPolynomialRegressor(
            regularization="ridge",
            alpha_ridge=alpha_ridge,  # type: ignore[arg-type]
        ).fit(x, y)


@pytest.mark.parametrize(
    "parameter",
    [
        "fit_intercept",
        "scale",
        "scale_y",
        "normalize_latent",
        "copy",
        "check_input",
    ],
)
@pytest.mark.parametrize(
    "estimator_class",
    [EIKGPolynomialRegressor, EIKGPolynomialRegressorCV],
)
def test_regressors_reject_non_boolean_parameters(
    parameter: str,
    estimator_class: type[EIKGPolynomialRegressor] | type[EIKGPolynomialRegressorCV],
) -> None:
    x, y = make_data(n=20)
    estimator = estimator_class(**{parameter: "False"})

    with pytest.raises(ValueError, match=parameter):
        estimator.fit(x, y)


def test_reproducible_results() -> None:
    x, y = make_data()
    m1 = EIKGPolynomialRegressor(degree=3).fit(x, y)
    m2 = EIKGPolynomialRegressor(degree=3).fit(x, y)
    p1 = m1.predict(x)
    p2 = m2.predict(x)
    np.testing.assert_allclose(p1, p2, rtol=1e-10, atol=1e-10)


def test_failed_refit_does_not_leave_mixed_fitted_state() -> None:
    x, y = make_data()
    model = EIKGPolynomialRegressor(regularization="ridge").fit(x, y)
    model.alpha_ridge = -1.0

    with pytest.raises(ValueError, match="alpha_ridge"):
        model.fit(10.0 * x, y)

    assert not getattr(model, "is_fitted_", False)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(x)


def test_refit_without_dataframe_drops_stale_feature_names() -> None:
    pd = pytest.importorskip("pandas")
    x, y = make_data()
    x_df = pd.DataFrame(x, columns=["a", "b", "c"])
    model = EIKGPolynomialRegressor().fit(x_df, y)
    assert hasattr(model, "feature_names_in_")

    model.fit(x, y)

    assert not hasattr(model, "feature_names_in_")


def test_score_rejects_target_length_mismatch() -> None:
    x, y = make_data()
    model = EIKGPolynomialRegressor().fit(x, y)

    with pytest.raises(ValueError, match="row mismatch"):
        model.score(x, np.ones(1))

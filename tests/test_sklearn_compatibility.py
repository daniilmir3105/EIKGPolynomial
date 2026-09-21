import numpy as np
import pytest

from eikg.regressors import EIKGPolynomialRegressor, EIKGPolynomialRegressorCV

sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone, is_regressor  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def make_data(seed: int = 11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(90, 4))
    z = 0.8 + 1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.1 * x[:, 2]
    y = z + 0.12 * z**2 + rng.normal(scale=0.05, size=z.shape[0])
    return x, y


def test_pipeline_compatibility() -> None:
    x, y = make_data()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", EIKGPolynomialRegressor(degree=2, scale=False)),
        ]
    )
    pipe.fit(x, y)
    pred = pipe.predict(x)
    assert pred.shape == (x.shape[0],)


def test_predict_before_fit_raises_sklearn_not_fitted_error() -> None:
    x, _ = make_data()

    with pytest.raises(NotFittedError, match="not fitted"):
        EIKGPolynomialRegressor().predict(x)


def test_grid_search_compatibility() -> None:
    x, y = make_data()
    model = EIKGPolynomialRegressor()
    grid = GridSearchCV(
        model,
        param_grid={"degree": [1, 2, 3], "regularization": ["none", "ridge"]},
        cv=3,
        scoring="neg_mean_squared_error",
    )
    grid.fit(x, y)
    assert grid.best_estimator_ is not None


def test_cv_regressor_selects_degree() -> None:
    x, y = make_data()
    model = EIKGPolynomialRegressorCV(max_degree=4, cv=3, scoring="neg_mean_squared_error")
    model.fit(x, y)
    assert 1 <= model.selected_degree_ <= 4


def test_existing_estimators_have_regressor_tags() -> None:
    assert is_regressor(EIKGPolynomialRegressor())
    assert is_regressor(EIKGPolynomialRegressorCV())


def test_cv_regressor_clone_preserves_inner_parameters() -> None:
    model = EIKGPolynomialRegressorCV(
        max_degree=3,
        cv=2,
        regularization="ridge",
        alpha_ridge=0.25,
        scale=False,
    )

    cloned = clone(model)

    assert cloned.get_params(deep=False) == model.get_params(deep=False)
    assert cloned.regularization == model.regularization
    assert cloned.alpha_ridge == model.alpha_ridge
    assert cloned.scale == model.scale

import numpy as np
import pytest

from eikg.regressors import EIKGPolynomialRegressor, EIKGPolynomialRegressorCV

sklearn = pytest.importorskip("sklearn")
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

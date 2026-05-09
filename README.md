# EIKGP Regressor

`EIKGPolynomialRegressor` is a compact two-stage regression model inspired by the elementary image of a Kolmogorov-Gabor polynomial:

1. Linear latent stage:
   `z = b0 + b1*x1 + ... + bm*xm`
2. Polynomial output stage:
   `y_hat = a0 + a1*z + a2*z^2 + ... + ad*z^d`

This implementation is built for numerical stability and sklearn-style workflows.

## Important warning

The model is a **compressed elementary image** of the Kolmogorov-Gabor polynomial and is **not equivalent** to direct estimation of the full multivariate polynomial basis.

## Installation

From PyPI (after release):

```bash
pip install eikgp-regressor
```

From source (editable):

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[dev]"
```

## Basic usage

```python
import numpy as np
from eikg import EIKGPolynomialRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 3))
z = 1.0 + 1.8 * X[:, 0] - 0.9 * X[:, 1]
y = z + 0.15 * z**2 + rng.normal(0, 0.1, size=200)

model = EIKGPolynomialRegressor(
    degree=3,
    regularization="ridge",
    alpha_ridge=1e-5,
    scale=True,
    normalize_latent=True,
)
model.fit(X, y)
pred = model.predict(X)
print(model.score(X, y))
```

## Pipeline example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eikg import EIKGPolynomialRegressor

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", EIKGPolynomialRegressor(scale=False, degree=2)),
    ]
)
pipe.fit(X, y)
```

## GridSearchCV example

```python
from sklearn.model_selection import GridSearchCV
from eikg import EIKGPolynomialRegressor

search = GridSearchCV(
    EIKGPolynomialRegressor(),
    param_grid={
        "degree": [1, 2, 3, 4],
        "regularization": ["none", "ridge"],
        "alpha_ridge": [1e-8, 1e-5, 1e-3],
    },
    scoring="neg_mean_squared_error",
    cv=5,
)
search.fit(X, y)
print(search.best_params_)
```

## Automatic degree selection

```python
from eikg import EIKGPolynomialRegressorCV

cv_model = EIKGPolynomialRegressorCV(
    max_degree=6,
    cv=5,
    scoring="neg_mean_squared_error",
    regularization="ridge",
    alpha_ridge=1e-5,
)
cv_model.fit(X, y)
print(cv_model.selected_degree_, cv_model.best_score_)
```

## Main limitations

- Model expressiveness is bounded by one latent linear projection.
- Very high `degree` can still be unstable without meaningful scaling/normalization.
- Quality of fit depends on whether target structure is well-approximated by polynomial-in-latent form.

## Development quality checks

```bash
ruff check .
mypy eikg
pytest
```

You can install git hooks for automatic local checks:

```bash
pre-commit install
pre-commit run --all-files
```

## Build and publish

Build distribution artifacts:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```


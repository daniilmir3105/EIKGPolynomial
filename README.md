# EIKGPolynomial

`EIKGPolynomial` is a lightweight Python package for regression models based on the **elementary image of the Kolmogorov-Gabor polynomial**.

The package currently provides two sklearn-style estimators:

* `EIKGPolynomialRegressor`
* `EIKGPolynomialRegressorCV`

The model uses a compact two-stage representation:

1. Linear latent stage:

```text
z = b0 + b1*x1 + ... + bm*xm
```

2. Polynomial output stage:

```text
y_hat = a0 + a1*z + a2*z^2 + ... + ad*z^d
```

This implementation is designed for numerically stable regression workflows with a compact polynomial-in-latent structure.

## Scientific background

The implementation is inspired by the concept of the elementary image of the Kolmogorov-Gabor polynomial proposed in:

> Svetunkov S. Elementary image of the Kolmogorov-Gabor polynomial in economic modeling. *Technoeconomics*, 2024, Vol. 3, No. 2 (9), pp. 4–21.
> DOI: https://doi.org/10.57809/2024.3.2.9.1

Original Russian title:

> Светуньков С. Элементарный образ полинома Колмогорова-Габора в моделировании экономики // Техноэкономика. 2024. Т. 3, № 2 (9). С. 4–21.

## Important note

`EIKGPolynomialRegressor` implements a **compressed elementary image** of the Kolmogorov-Gabor polynomial.

It is not equivalent to direct estimation of the full multivariate polynomial basis. Instead of constructing all multivariate polynomial terms explicitly, the model first builds one latent linear projection and then applies a univariate polynomial transformation to this latent variable.

## Installation

Install the current released version from PyPI:

```bash
pip install eikgp-regressor
```

Install from source in editable mode:

```bash
git clone https://github.com/daniilmir3105/EIKGPolynomial.git
cd EIKGPolynomial
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Optional dependencies can be installed separately when needed:

```bash
pip install -e ".[scipy]"
pip install -e ".[sklearn]"
pip install -e ".[pandas]"
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

y_pred = model.predict(X)
r2 = model.score(X, y)

print(r2)
```

## `EIKGPolynomialRegressor`

`EIKGPolynomialRegressor` is the main regression model.

```python
from eikg import EIKGPolynomialRegressor
```

### Main parameters

| Parameter          |                        Type |      Default | Description                                                                                   |
| ------------------ | --------------------------: | -----------: | --------------------------------------------------------------------------------------------- |
| `degree`           |                       `int` |          `2` | Degree of the polynomial transformation applied to the latent variable.                       |
| `regularization`   | `"none"`, `"ridge"`, `None` |     `"none"` | Least-squares regularization mode.                                                            |
| `alpha_ridge`      |                     `float` |       `1e-8` | Ridge regularization strength. Used only when `regularization="ridge"`.                       |
| `fit_intercept`    |                      `bool` |       `True` | Whether to fit intercept terms in both stages.                                                |
| `scale`            |                      `bool` |       `True` | Whether to standardize input features `X` before fitting.                                     |
| `scale_y`          |                      `bool` |      `False` | Whether to standardize target values `y` during fitting and invert scaling during prediction. |
| `normalize_latent` |                      `bool` |       `True` | Whether to normalize the latent variable before building polynomial powers.                   |
| `dtype`            |         NumPy floating type | `np.float64` | Internal numeric dtype.                                                                       |
| `copy`             |                      `bool` |       `True` | Whether to copy input arrays during validation.                                               |
| `check_input`      |                      `bool` |       `True` | Whether to check input shapes and finite values.                                              |
| `lstsq_rcond`      |           `float` or `None` |       `None` | Cutoff parameter passed to the least-squares solver.                                          |

### Practical recommendations

For most tasks, start with:

```python
model = EIKGPolynomialRegressor(
    degree=2,
    regularization="ridge",
    alpha_ridge=1e-5,
    scale=True,
    normalize_latent=True,
)
```

Use `regularization="ridge"` when the data are noisy, the features are correlated, or the polynomial degree is greater than 2.

Keep `scale=True` unless the input data are already standardized.

Keep `normalize_latent=True` for better numerical stability, especially for higher polynomial degrees.

Use `scale_y=True` when target values have a large magnitude or a very wide numerical range.

## `EIKGPolynomialRegressorCV`

`EIKGPolynomialRegressorCV` automatically selects the polynomial degree using cross-validation.

```python
from eikg import EIKGPolynomialRegressorCV
```

Example:

```python
cv_model = EIKGPolynomialRegressorCV(
    max_degree=6,
    cv=5,
    scoring="neg_mean_squared_error",
    regularization="ridge",
    alpha_ridge=1e-5,
    scale=True,
    normalize_latent=True,
)

cv_model.fit(X, y)

print(cv_model.selected_degree_)
print(cv_model.best_score_)
```

### Main parameters

| Parameter            |                               Type |                    Default | Description                                                |
| -------------------- | ---------------------------------: | -------------------------: | ---------------------------------------------------------- |
| `max_degree`         |                              `int` |                        `6` | Maximum degree tested during cross-validation.             |
| `cv`                 |                              `int` |                        `5` | Number of cross-validation folds.                          |
| `scoring`            | `"neg_mean_squared_error"`, `"r2"` | `"neg_mean_squared_error"` | Metric used for degree selection.                          |
| `**regressor_kwargs` |                  keyword arguments |                          — | Additional parameters passed to `EIKGPolynomialRegressor`. |

For example, these parameters are passed directly to the inner regressor:

```python
cv_model = EIKGPolynomialRegressorCV(
    max_degree=5,
    regularization="ridge",
    alpha_ridge=1e-4,
    scale=True,
    normalize_latent=True,
)
```

## Fitted attributes

After calling `fit`, the model stores learned parameters and diagnostics:

| Attribute           | Description                                        |
| ------------------- | -------------------------------------------------- |
| `beta_`             | Coefficients of the first linear latent stage.     |
| `intercept_1_`      | Intercept of the first stage.                      |
| `alpha_`            | Coefficients of the polynomial output stage.       |
| `intercept_2_`      | Intercept of the second stage.                     |
| `degree_`           | Fitted polynomial degree.                          |
| `n_features_in_`    | Number of input features seen during fitting.      |
| `condition_number_` | Condition number of the first-stage design matrix. |
| `rank_`             | Rank of the first-stage design matrix.             |
| `rank_latent_`      | Rank of the latent polynomial design matrix.       |

For `EIKGPolynomialRegressorCV`, the main fitted attributes are:

| Attribute          | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `selected_degree_` | Degree selected by cross-validation.                 |
| `best_score_`      | Best cross-validation score.                         |
| `cv_scores_`       | Mean cross-validation scores for all tested degrees. |
| `estimator_`       | Final fitted `EIKGPolynomialRegressor`.              |

## Minimal API

Both estimators follow the standard sklearn-style workflow:

```python
model.fit(X, y)
y_pred = model.predict(X)
score = model.score(X, y)
```

`score(X, y)` returns the coefficient of determination `R^2`.

## Limitations

* The model uses one latent linear projection, so its expressiveness is more limited than a full multivariate polynomial model.
* High polynomial degrees may be numerically unstable without scaling, latent normalization, or regularization.
* The model is most suitable when the target can be reasonably approximated by a polynomial function of a compact latent representation.

## Development checks

```bash
ruff check .
mypy eikg
pytest
```

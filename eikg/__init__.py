"""EIKG regressors for compact Kolmogorov-Gabor modeling."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .networks import (
    CombinatorialPolynomialNetwork,
    DeepPolyNetwork,
    DeepPolyNetworkCV,
    PolynomialNetwork,
    PolynomialNetworkCV,
)
from .regressors import EIKGPolynomialRegressor, EIKGPolynomialRegressorCV

try:
    __version__ = _distribution_version("eikgpolynomial")
except _PackageNotFoundError:
    # The distribution metadata is absent when the package is imported directly
    # from an unpacked source tree that has not been installed.
    __version__ = "0.2.0"

__all__ = [
    "__version__",
    "EIKGPolynomialRegressor",
    "EIKGPolynomialRegressorCV",
    "CombinatorialPolynomialNetwork",
    "DeepPolyNetwork",
    "DeepPolyNetworkCV",
    "PolynomialNetwork",
    "PolynomialNetworkCV",
]

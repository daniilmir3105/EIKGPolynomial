# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and the project adheres to Semantic Versioning.

## [0.1.1] - 2026-05-09

### Changed
- Renamed distribution package from `eikg-regressor` to `eikgp-regressor`.
- Added release and publishing guidance for repeatable pip/TestPyPI/PyPI workflow.
- Finalized CI + lint/type/test/build validation for publish-ready state.

## [0.1.0] - 2026-05-09

### Added
- Initial `eikg` package structure and public API.
- `EIKGPolynomialRegressor` with numerically stable two-stage fitting.
- `EIKGPolynomialRegressorCV` for automatic degree selection via cross-validation.
- Optional support for `pandas`, `scipy`, and `scikit-learn`.
- Strict input validation utilities and lightweight preprocessing/metrics modules.
- Pytest suite for regressor behavior, validation, and sklearn compatibility.
- README with usage, pipeline, and GridSearch examples.
- Packaging metadata for pip installation and publishing.


# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and the project adheres to Semantic Versioning.

## [Unreleased]

## [0.2.0] - 2026-09-21

### Added

- `DeepPolyNetwork`, a fixed-width cascade of independently fitted
  `EIKGPolynomialRegressor` layers with an explicit `n_layers` parameter.
- Canonical `DeepPolyNetwork` and `DeepPolyNetworkCV` public names; the former
  `PolynomialNetwork` and `PolynomialNetworkCV` names remain backward-compatible aliases.
- Train-only max-absolute scaling for original features and intermediate predictions before
  constructing successive layer inputs.
- Scalar or per-layer degree configurations for `DeepPolyNetwork`, with canonical fitted
  `degrees_` metadata.
- `DeepPolyNetworkCV` for leakage-safe greedy selection of a separate degree at every layer,
  fold-local prefix construction, and a final refit on all supplied training data.
- `CombinatorialPolynomialNetwork`, a two-level estimator that evaluates bounded feature
  combinations with `EIKGPolynomialRegressorCV`, ranks them by ascending mean CV MSE, retains
  Top-K, and fits a final polynomial learner on their post-selection OOF predictions.
- Configurable `min_combination_size`, `max_combination_size`, and a hard pre-fit
  `max_candidates` cap to prevent an implicit combinatorial search explosion.
- Combinatorial-network diagnostics including `ranking_`, `selected_combinations_`,
  `selected_models_`, `selected_degrees_`, `oof_predictions_`, `final_estimator_`, and
  `final_degree_`.
- Per-layer CV diagnostics: `selected_degrees_`, `layer_cv_scores_`,
  `layer_cv_fold_scores_`, and `layer_best_scores_`.
- Validation and diagnostics for non-finite inputs, unstable powered features, fitted state,
  repeated fitting, and feature-name consistency.
- Documentation for the network mathematics, API, numerical-stability behavior,
  cross-validation design, computational cost, and known limitations.
- Documentation for candidate enumeration, mean-CV-MSE ranking, Top-K selection,
  post-selection OOF semantics, the need for outer CV when estimating the whole combinatorial
  procedure, and the reason no separate combinatorial CV class is provided.
- Release automation for Python 3.10-3.14, built-wheel smoke testing, and tokenless PyPI
  publication through GitHub Actions Trusted Publishing.
- Source distributions now include the changelog and release guide.
- A public `eikg.__version__` value backed by installed distribution metadata, with a source-tree
  fallback when package metadata is unavailable.

### Changed

- Renamed the PyPI distribution from `eikgp-regressor` to `eikgpolynomial`. The import package
  remains `eikg`. This release is the first upload of the new name and does not publish the
  former distribution name.
- Replaced the Ridge normal-equation solve with a thin-SVD filter to avoid squaring the design
  matrix condition number, avoid a dense penalty identity, and support rank-deficient inputs,
  including `alpha_ridge=0`.
- Added validation requiring `alpha_ridge` to be finite and non-negative.
- Restored modern scikit-learn regressor tags for the existing estimators and made
  `EIKGPolynomialRegressorCV` expose its layer options as explicit, clone-safe parameters.
- `EIKGPolynomialRegressorCV` now retains per-degree fold scores and selected-degree OOF
  predictions so stacking estimators can reuse the same fold evaluations without in-sample
  first-level predictions.
- Fitted state is now cleared on refit, preventing stale DataFrame metadata or mixed model/scaler
  state after an unsuccessful repeated fit.
- Lightweight scaling now raises diagnostic errors for non-finite statistics and transformed
  values instead of allowing overflow warnings to become corrupted model inputs.
- Scaling and latent normalization use max-abs-preconditioned statistics so very large finite
  targets can be processed without overflowing mean or variance calculations.
- Complex-valued inputs and unsupported numeric dtypes are rejected before lossy conversion;
  scoring validates sample counts and follows the standard constant-target R-squared convention.
- Legacy network-CV diagnostics (`selected_degree_`, `cv_scores_`, `cv_fold_scores_`, and
  `best_score_`) remain available and now explicitly describe only the final greedy selection
  step; complete per-layer results are exposed through the new diagnostics.
- Release documentation now distinguishes import aliases from historical behavioral
  compatibility, records the supported scikit-learn integration surface, and requires a fresh
  exact-commit build before publication.

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


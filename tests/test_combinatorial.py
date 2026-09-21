from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from eikg import CombinatorialPolynomialNetwork
from eikg.metrics import mean_squared_error
from eikg.regressors import (
    EIKGPolynomialRegressor,
    EIKGPolynomialRegressorCV,
    NotFittedError,
)

LAYER_PARAMETERS: dict[str, Any] = {
    "regularization": "ridge",
    "alpha_ridge": 1e-5,
    "fit_intercept": True,
    "scale": True,
    "scale_y": False,
    "normalize_latent": True,
    "dtype": np.float64,
    "copy": True,
    "check_input": True,
    "lstsq_rcond": None,
}


def make_data(
    n_samples: int = 36, n_features: int = 4, seed: int = 801
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    weights = np.linspace(1.3, -0.25, n_features)
    latent = 0.7 + x @ weights
    y = latent + 0.12 * latent**2 + rng.normal(scale=0.02, size=n_samples)
    return x, y


def contiguous_kfold_indices(
    n_samples: int, n_splits: int
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    all_indices = np.arange(n_samples)
    current = 0
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        validation_indices = all_indices[start:stop]
        train_indices = np.concatenate((all_indices[:start], all_indices[stop:]))
        splits.append((train_indices, validation_indices))
        current = stop
    return splits


def enumerate_combinations(
    n_features: int, min_size: int, max_size: int
) -> list[tuple[int, ...]]:
    return [
        combination
        for size in range(min_size, max_size + 1)
        for combination in combinations(range(n_features), size)
    ]


def manual_candidate_ranking(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    min_size: int,
    max_size: int,
    max_degree: int,
    cv: int,
) -> list[dict[str, Any]]:
    """Rank candidates with fixed-degree fold models, independently of the wrapper."""

    splits = contiguous_kfold_indices(x.shape[0], cv)
    records: list[dict[str, Any]] = []
    for combination in enumerate_combinations(x.shape[1], min_size, max_size):
        results_by_degree: list[tuple[float, list[float], NDArray[np.float64]]] = []
        for degree in range(1, max_degree + 1):
            fold_mse: list[float] = []
            oof_predictions = np.empty(x.shape[0], dtype=np.float64)
            for train_indices, validation_indices in splits:
                estimator = EIKGPolynomialRegressor(
                    degree=degree,
                    **LAYER_PARAMETERS,
                ).fit(x[train_indices][:, combination], y[train_indices])
                prediction = estimator.predict(x[validation_indices][:, combination])
                oof_predictions[validation_indices] = prediction
                fold_mse.append(mean_squared_error(y[validation_indices], prediction))
            results_by_degree.append((float(np.mean(fold_mse)), fold_mse, oof_predictions))

        best_index = int(np.argmin([result[0] for result in results_by_degree]))
        mse_mean, mse_folds, oof_predictions = results_by_degree[best_index]
        records.append(
            {
                "combination": combination,
                "combination_size": len(combination),
                "selected_degree": best_index + 1,
                "cv_mse_mean": mse_mean,
                "cv_mse_std": float(np.std(np.asarray(mse_folds, dtype=np.float64))),
                "cv_mse_fold_scores": tuple(mse_folds),
                "oof_predictions": oof_predictions,
            }
        )

    return sorted(records, key=lambda record: (record["cv_mse_mean"], record["combination"]))


@pytest.fixture(scope="module")
def ranked_case() -> tuple[
    CombinatorialPolynomialNetwork,
    NDArray[np.float64],
    NDArray[np.float64],
    list[dict[str, Any]],
]:
    x, y = make_data()
    model = CombinatorialPolynomialNetwork(
        top_k=3,
        min_combination_size=2,
        max_combination_size=2,
        max_candidates=10,
        max_degree=2,
        cv=3,
        **LAYER_PARAMETERS,
    ).fit(x, y)
    expected_ranking = manual_candidate_ranking(
        x,
        y,
        min_size=2,
        max_size=2,
        max_degree=2,
        cv=3,
    )
    return model, x, y, expected_ranking


def test_combinatorial_constructor_defaults_and_unfitted_state() -> None:
    model = CombinatorialPolynomialNetwork()

    assert model.top_k == 5
    assert model.min_combination_size == 2
    assert model.max_combination_size == 2
    assert model.max_candidates == 1000
    assert model.max_degree == 6
    assert model.cv == 5
    assert not hasattr(model, "ranking_")

    x, y = make_data(n_samples=12)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(x)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.score(x, y)


@pytest.mark.parametrize("top_k", [1, 5, 10, 20])
def test_combinatorial_supports_required_top_k_values(top_k: int) -> None:
    x, y = make_data(n_samples=20, n_features=7, seed=802)
    model = CombinatorialPolynomialNetwork(
        top_k=top_k,
        max_candidates=32,
        max_degree=1,
        cv=2,
        **LAYER_PARAMETERS,
    ).fit(x, y)

    assert model.n_candidates_ == 21
    assert len(model.ranking_) == 21
    assert len(model.selected_combinations_) == top_k
    assert len(model.selected_models_) == top_k
    assert len(model.selected_degrees_) == top_k
    assert model.final_estimator_.n_features_in_ == top_k
    assert model.final_degree_ == model.final_estimator_.selected_degree_
    assert model.oof_predictions_.shape == (x.shape[0], top_k)
    assert sum(bool(record["selected"]) for record in model.ranking_) == top_k
    assert tuple(model.selected_combinations_) == tuple(
        tuple(record["combination"]) for record in model.ranking_[:top_k]
    )
    prediction = model.predict(x)
    assert prediction.shape == (x.shape[0],)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "5"])
def test_combinatorial_rejects_invalid_top_k(top_k: Any) -> None:
    x, y = make_data(n_samples=16)

    with pytest.raises((TypeError, ValueError), match="top_k"):
        CombinatorialPolynomialNetwork(top_k=top_k, max_degree=1, cv=2).fit(x, y)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"min_combination_size": 0}, "min_combination_size"),
        ({"min_combination_size": True}, "min_combination_size"),
        ({"min_combination_size": 3, "max_combination_size": 2}, "combination_size"),
        ({"max_combination_size": 5}, "features|combination_size"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_candidates": True}, "max_candidates"),
        ({"max_candidates": 5.5}, "max_candidates"),
    ],
)
def test_combinatorial_rejects_invalid_combination_parameters(
    parameters: dict[str, Any], message: str
) -> None:
    x, y = make_data(n_samples=16)

    with pytest.raises((TypeError, ValueError), match=message):
        CombinatorialPolynomialNetwork(
            top_k=1,
            max_degree=1,
            cv=2,
            **parameters,
        ).fit(x, y)


def test_combinatorial_enforces_candidate_cap_and_top_k_bound() -> None:
    x, y = make_data(n_samples=20)

    capped = CombinatorialPolynomialNetwork(
        top_k=1,
        max_candidates=5,
        max_degree=1,
        cv=2,
    )
    with pytest.raises(ValueError, match="max_candidates|candidate"):
        capped.fit(x, y)
    assert not hasattr(capped, "ranking_")

    with pytest.raises(ValueError, match="top_k|candidate"):
        CombinatorialPolynomialNetwork(
            top_k=7,
            max_candidates=6,
            max_degree=1,
            cv=2,
        ).fit(x, y)


def test_combinatorial_enumerates_requested_size_range() -> None:
    x, y = make_data(n_samples=20, n_features=5)
    model = CombinatorialPolynomialNetwork(
        top_k=1,
        min_combination_size=2,
        max_combination_size=3,
        max_candidates=20,
        max_degree=1,
        cv=2,
        **LAYER_PARAMETERS,
    ).fit(x, y)

    expected = enumerate_combinations(5, 2, 3)
    actual = [tuple(record["combination"]) for record in model.ranking_]
    assert model.n_candidates_ == 20
    assert len(actual) == len(set(actual)) == 20
    assert set(actual) == set(expected)
    assert {record["combination_size"] for record in model.ranking_} == {2, 3}


def test_combinatorial_ranking_matches_manual_cv_mse(
    ranked_case: tuple[
        CombinatorialPolynomialNetwork,
        NDArray[np.float64],
        NDArray[np.float64],
        list[dict[str, Any]],
    ],
) -> None:
    model, _, _, expected_ranking = ranked_case

    assert model.n_candidates_ == len(expected_ranking) == 6
    assert len(model.ranking_) == 6
    required_fields = {
        "rank",
        "combination",
        "feature_names",
        "combination_size",
        "selected_degree",
        "cv_mse_mean",
        "cv_mse_std",
        "cv_mse_fold_scores",
        "selected",
    }
    for index, (actual, expected) in enumerate(
        zip(model.ranking_, expected_ranking, strict=True), start=1
    ):
        assert required_fields <= actual.keys()
        assert actual["rank"] == index
        assert tuple(actual["combination"]) == expected["combination"]
        assert actual["combination_size"] == expected["combination_size"]
        assert actual["selected_degree"] == expected["selected_degree"]
        assert actual["cv_mse_mean"] == pytest.approx(expected["cv_mse_mean"])
        assert actual["cv_mse_std"] == pytest.approx(expected["cv_mse_std"])
        np.testing.assert_allclose(
            actual["cv_mse_fold_scores"],
            expected["cv_mse_fold_scores"],
            rtol=1e-11,
            atol=1e-11,
        )
        assert bool(actual["selected"]) is (index <= model.top_k)

    expected_selected = expected_ranking[: model.top_k]
    assert tuple(model.selected_combinations_) == tuple(
        record["combination"] for record in expected_selected
    )
    assert tuple(model.selected_degrees_) == tuple(
        record["selected_degree"] for record in expected_selected
    )


def test_combinatorial_uses_selected_candidate_oof_columns_for_final_fit(
    ranked_case: tuple[
        CombinatorialPolynomialNetwork,
        NDArray[np.float64],
        NDArray[np.float64],
        list[dict[str, Any]],
    ],
) -> None:
    model, _, y, expected_ranking = ranked_case
    expected_selected = expected_ranking[: model.top_k]
    expected_oof = np.column_stack(
        [record["oof_predictions"] for record in expected_selected]
    )
    stored_oof = np.column_stack(
        [candidate.oof_predictions_ for candidate in model.selected_models_]
    )

    # These are post-selection OOF columns: each row is self-target-free, while
    # global degree and Top-K selection still use all fold scores.
    np.testing.assert_allclose(stored_oof, expected_oof, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(model.oof_predictions_, expected_oof, rtol=1e-11, atol=1e-11)
    assert model.oof_predictions_.shape == (y.shape[0], model.top_k)
    assert np.isfinite(stored_oof).all()

    expected_final = EIKGPolynomialRegressorCV(
        max_degree=2,
        scoring="neg_mean_squared_error",
        cv=3,
        **LAYER_PARAMETERS,
    ).fit(expected_oof, y)
    assert model.final_degree_ == expected_final.selected_degree_
    np.testing.assert_allclose(
        model.final_estimator_.predict(expected_oof),
        expected_final.predict(expected_oof),
        rtol=1e-11,
        atol=1e-11,
    )


def test_combinatorial_oof_rows_do_not_use_their_own_fold_targets() -> None:
    x, y = make_data(n_samples=18, seed=806)
    parameters = {
        "top_k": 6,
        "max_candidates": 6,
        "max_degree": 1,
        "cv": 3,
        **LAYER_PARAMETERS,
    }
    baseline = CombinatorialPolynomialNetwork(**parameters).fit(x, y)
    changed_y = y.copy()
    first_validation_fold = np.arange(6)
    changed_y[first_validation_fold] += 50.0
    changed = CombinatorialPolynomialNetwork(**parameters).fit(x, changed_y)

    baseline_oof = {
        tuple(combination): baseline.oof_predictions_[:, column]
        for column, combination in enumerate(baseline.selected_combinations_)
    }
    changed_oof = {
        tuple(combination): changed.oof_predictions_[:, column]
        for column, combination in enumerate(changed.selected_combinations_)
    }
    assert baseline_oof.keys() == changed_oof.keys()
    for combination in baseline_oof:
        np.testing.assert_allclose(
            baseline_oof[combination][first_validation_fold],
            changed_oof[combination][first_validation_fold],
            rtol=0.0,
            atol=0.0,
        )


def test_combinatorial_prediction_matches_saved_two_level_chain(
    ranked_case: tuple[
        CombinatorialPolynomialNetwork,
        NDArray[np.float64],
        NDArray[np.float64],
        list[dict[str, Any]],
    ],
) -> None:
    model, x, _, _ = ranked_case
    candidate_predictions = np.column_stack(
        [
            candidate.predict(x[:, combination])
            for combination, candidate in zip(
                model.selected_combinations_, model.selected_models_, strict=True
            )
        ]
    )
    expected = model.final_estimator_.predict(candidate_predictions)

    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-11, atol=1e-11)


def test_combinatorial_repeat_fit_replaces_all_state() -> None:
    x_first, y_first = make_data(n_samples=24, n_features=4, seed=811)
    x_second, y_second = make_data(n_samples=28, n_features=5, seed=812)
    parameters = {
        "top_k": 2,
        "max_candidates": 16,
        "max_degree": 1,
        "cv": 2,
        **LAYER_PARAMETERS,
    }
    model = CombinatorialPolynomialNetwork(**parameters).fit(x_first, y_first)
    old_models = tuple(model.selected_models_)
    old_final = model.final_estimator_

    model.fit(x_second, y_second)
    fresh = CombinatorialPolynomialNetwork(**parameters).fit(x_second, y_second)

    assert model.n_candidates_ == 10
    assert len(model.ranking_) == 10
    assert all(new is not old for new in model.selected_models_ for old in old_models)
    assert model.final_estimator_ is not old_final
    np.testing.assert_allclose(
        model.predict(x_second), fresh.predict(x_second), rtol=1e-11, atol=1e-11
    )


def test_combinatorial_failed_refit_clears_fitted_state() -> None:
    x, y = make_data(n_samples=24)
    model = CombinatorialPolynomialNetwork(
        top_k=2,
        max_candidates=8,
        max_degree=1,
        cv=2,
    ).fit(x, y)
    bad_x = x.copy()
    bad_x[0, 0] = np.nan

    with pytest.raises(ValueError, match="X contains"):
        model.fit(bad_x, y)

    assert not hasattr(model, "ranking_")
    assert not hasattr(model, "final_estimator_")
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(x)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_combinatorial_rejects_nonfinite_target(bad_value: float) -> None:
    x, y = make_data(n_samples=16)
    bad_y = y.copy()
    bad_y[0] = bad_value

    with pytest.raises(ValueError, match="y contains"):
        CombinatorialPolynomialNetwork(
            top_k=1,
            max_candidates=8,
            max_degree=1,
            cv=2,
        ).fit(x, bad_y)


def test_combinatorial_validates_cv_predict_features_and_score_length() -> None:
    x, y = make_data(n_samples=16)

    with pytest.raises(ValueError, match="cv|samples"):
        CombinatorialPolynomialNetwork(
            top_k=1,
            max_candidates=8,
            max_degree=1,
            cv=x.shape[0] + 1,
        ).fit(x, y)

    model = CombinatorialPolynomialNetwork(
        top_k=2,
        max_candidates=8,
        max_degree=1,
        cv=2,
    ).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(x[:, :3])
    with pytest.raises(ValueError, match="row mismatch"):
        model.score(x, np.ones(1))


def test_combinatorial_dataframe_names_order_and_refit() -> None:
    pd = pytest.importorskip("pandas")
    x, y = make_data(n_samples=24)
    columns = ["alpha", "beta value", "gamma", "delta"]
    index = np.arange(x.shape[0]) * 11 + 100
    x_df = pd.DataFrame(x, columns=columns, index=index)
    y_series = pd.Series(y, index=index)
    original_x = x_df.copy(deep=True)
    original_y = y_series.copy(deep=True)
    model = CombinatorialPolynomialNetwork(
        top_k=2,
        max_candidates=8,
        max_degree=1,
        cv=2,
        **LAYER_PARAMETERS,
    ).fit(x_df, y_series)

    np.testing.assert_array_equal(model.feature_names_in_, np.asarray(columns, dtype=str))
    for record in model.ranking_:
        expected_names = tuple(columns[index] for index in record["combination"])
        assert tuple(record["feature_names"]) == expected_names
    assert np.isfinite(model.predict(x_df)).all()
    pd.testing.assert_frame_equal(x_df, original_x)
    pd.testing.assert_series_equal(y_series, original_y)

    with pytest.raises(ValueError, match="columns|feature names"):
        model.predict(x_df[["beta value", "alpha", "gamma", "delta"]])

    model.fit(x, y)
    assert not hasattr(model, "feature_names_in_")
    assert all(record["feature_names"] is None for record in model.ranking_)


def test_combinatorial_is_stable_for_large_finite_features() -> None:
    rng = np.random.default_rng(821)
    magnitude = 1e100
    x = rng.normal(size=(28, 4)) * magnitude
    scaled = x / magnitude
    y = 0.5 + 1.1 * scaled[:, 0] - 0.6 * scaled[:, 1] + 0.08 * scaled[:, 0] ** 2
    model = CombinatorialPolynomialNetwork(
        top_k=2,
        max_candidates=8,
        max_degree=2,
        cv=2,
        scale_y=True,
        regularization="ridge",
        alpha_ridge=1e-5,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        model.fit(x, y)
        prediction = model.predict(x * 0.9)

    assert np.isfinite(prediction).all()
    assert all(np.isfinite(record["cv_mse_mean"]) for record in model.ranking_)
    assert all(np.isfinite(record["cv_mse_std"]) for record in model.ranking_)


def test_combinatorial_is_sklearn_cloneable() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone, is_regressor

    del sklearn
    model = CombinatorialPolynomialNetwork(
        top_k=3,
        min_combination_size=1,
        max_combination_size=3,
        max_candidates=50,
        max_degree=2,
        cv=3,
        regularization="none",
        alpha_ridge=0.25,
        fit_intercept=False,
        scale=False,
        scale_y=True,
        normalize_latent=False,
        dtype=np.float32,
        copy=False,
        check_input=False,
        lstsq_rcond=1e-7,
    )
    cloned = clone(model)

    assert cloned.get_params(deep=False) == model.get_params(deep=False)
    assert is_regressor(cloned)
    for attribute in (
        "ranking_",
        "n_candidates_",
        "selected_combinations_",
        "selected_models_",
        "selected_degrees_",
        "oof_predictions_",
        "final_estimator_",
        "final_degree_",
    ):
        assert not hasattr(cloned, attribute)


def test_combinatorial_works_in_sklearn_pipeline() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    del sklearn
    x, y = make_data(n_samples=28)
    pipeline = Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "network",
                CombinatorialPolynomialNetwork(
                    top_k=2,
                    max_candidates=8,
                    max_degree=1,
                    cv=2,
                    regularization="ridge",
                    alpha_ridge=1e-5,
                    scale=False,
                ),
            ),
        ]
    )

    pipeline.fit(x, y)
    prediction = pipeline.predict(x)

    assert prediction.shape == (x.shape[0],)
    assert np.isfinite(prediction).all()


def test_combinatorial_works_in_sklearn_grid_search() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.model_selection import GridSearchCV

    del sklearn
    x, y = make_data(n_samples=20, seed=831)
    search = GridSearchCV(
        CombinatorialPolynomialNetwork(
            top_k=1,
            max_candidates=8,
            max_degree=1,
            cv=2,
            regularization="ridge",
            alpha_ridge=1e-5,
        ),
        param_grid={"top_k": [1, 2]},
        cv=2,
        scoring="neg_mean_squared_error",
    )

    search.fit(x, y)
    prediction = search.predict(x)

    assert search.best_params_["top_k"] in {1, 2}
    assert search.best_estimator_.is_fitted_ is True
    assert prediction.shape == (x.shape[0],)
    assert np.isfinite(prediction).all()

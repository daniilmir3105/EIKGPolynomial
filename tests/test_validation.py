import numpy as np
import pytest

from eikg.validation import validate_degree, validate_x, validate_xy_lengths, validate_y


def test_validate_degree_raises_for_invalid() -> None:
    with pytest.raises(ValueError, match="degree must be >="):
        validate_degree(0)


def test_validate_xy_length_mismatch() -> None:
    x = np.ones((5, 2))
    y = np.ones(4)
    with pytest.raises(ValueError, match="row mismatch"):
        validate_xy_lengths(x, y)


def test_validate_x_nan_raises() -> None:
    x = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="contains NaN"):
        validate_x(x)


def test_validate_y_multicolumn_df_raises() -> None:
    pd = pytest.importorskip("pandas")
    y = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="exactly one column"):
        validate_y(y)

"""
"""
import pytest

from cascade.dismod.db.wrapper import DismodFile
from cascade.executor.model_residuals_main import _get_residuals


@pytest.fixture
def dm_file_ok():
    dm_file_ok = None
    return dm_file_ok


@pytest.fixture
def dm_file_invalid():
    dm_file_invalid = None
    return dm_file_invalid


@pytest.fixture
def dm_file_empty():
    dm_file_empty = DismodFile()
    dm_file_empty.empty_table("fit_var")
    dm_file_empty.empty_table("fit_data_subset")
    return dm_file_empty


def test_get_model_residuals_inputs_ok(ihme, dm_file_ok):
    """
    """
    fv_residuals_columns = ["residual_value", "residual_dage", "residual_dtime"]
    fds_residuals_columns = ["weighted_residual"]

    fv_residuals = _get_residuals(dm_file_ok)
    fds_residuals = _get_residuals(dm_file_ok)

    assert set(fv_residuals.columns) == set(fv_residuals_columns)
    assert set(fds_residuals.columns) == set(fds_residuals_columns)

    # at_row_index_8 = pd.Series([265844, 1990, 90, 1, 2, 16, 0.161961, 0.161961, 0.161961])
    # at_row_index_8.index = at_results.columns

    # pd.testing.assert_series_equal(at_results.iloc[8], at_row_index_8,
    #                                   check_exact=False, check_names=False)


def test_get_model_residuals_bad_dismod_file(ihme, dm_file_invalid):
    """Expect an exception if dismod file is not valid"""
    with pytest.raises(ValueError):
        _get_residuals(dm_file_invalid)


def test_get_model_residuals_empty_dismod_file(ihme, dm_file_empty):
    """Expect an empty dataframe if the dismod file has empty fit_var and fit_data_subset tables"""

    fv_residuals, fds_residuals = _get_residuals(dm_file_empty)

    assert fv_residuals.empty
    assert fds_residuals.empty

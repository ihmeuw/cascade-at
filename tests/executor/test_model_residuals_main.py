"""Test stand-alone program which extracts residuals data from a Dismod file.
"""
import pytest

import pandas as pd

from cascade.dismod.db.wrapper import DismodFile
from cascade.executor.model_residuals_main import _get_residuals


@pytest.fixture
def dm_file_ok():
    dm_file_ok = DismodFile()

    fit_var = pd.DataFrame()
    fit_var["fit_var_id"] = [0, 1, 2, 3]
    fit_var["variable_value"] = [0.0, -3.6, -4.1, -4.0]
    fit_var["residual_value"] = [None, -18.0, -20.6, -20.1]
    fit_var["residual_dage"] = [None, None, None, None]
    fit_var["residual_dtime"] = [None, None, None, None]
    fit_var["lagrange_value"] = [581.7, 0.0, 0.0, 0.0]
    fit_var["lagrange_dage"] = [0.0, 0.0, 0.0, 0.0]
    fit_var["lagrange_dtime"] = [0.0, 0.0, 0.0, 0.0]

    dm_file_ok.fit_var = fit_var

    fit_data_subset = pd.DataFrame()
    fit_data_subset["fit_data_subset_id"] = [0, 1, 2, 3]
    fit_data_subset["avg_integrand"] = [0.021, 0.022, 0.022, 0.022]
    fit_data_subset["weighted_residual"] = [-250.9, -201.4, -65.9, -74.1]

    dm_file_ok.fit_data_subset = fit_data_subset

    return dm_file_ok


@pytest.fixture
def dm_file_invalid():
    dm_file_invalid = DismodFile()
    return dm_file_invalid


@pytest.fixture
def dm_file_empty():
    dm_file_empty = DismodFile()
    dm_file_empty.empty_table("fit_var")
    dm_file_empty.empty_table("fit_data_subset")
    return dm_file_empty


def test_get_model_residuals_inputs_ok(dm_file_ok):
    """dm_file has the expected tables.
    """
    fv_residuals_columns = ["fit_var_id", "variable_value", "residual_value", "residual_dage",
                            "residual_dtime", "lagrange_value", "lagrange_dage", "lagrange_dtime"]
    fds_residuals_columns = ["fit_data_subset_id", "avg_integrand", "weighted_residual"]

    fv_residuals, fds_residuals = _get_residuals(dm_file_ok)

    assert set(fv_residuals.columns) == set(fv_residuals_columns)
    assert set(fds_residuals.columns) == set(fds_residuals_columns)

    fv_residuals_residual_value = pd.Series([None, -18.0, -20.6, -20.1])

    pd.testing.assert_series_equal(fv_residuals["residual_value"], fv_residuals_residual_value,
                                   check_exact=False, check_names=False)

    fds_residuals_weighted_residual = pd.Series([-250.9, -201.4, -65.9, -74.1])

    pd.testing.assert_series_equal(fds_residuals["weighted_residual"], fds_residuals_weighted_residual,
                                   check_exact=False, check_names=False)


def test_get_model_residuals_bad_dismod_file(dm_file_invalid):
    """Expect empty dataframes if the dismod file has no fit_var and fit_data_subset tables"""

    fv_residuals, fds_residuals = _get_residuals(dm_file_invalid)

    assert fv_residuals.empty
    assert fds_residuals.empty


def test_get_model_residuals_empty_dismod_file(dm_file_empty):
    """Expect empty dataframes if the dismod file has empty fit_var and fit_data_subset tables"""

    fv_residuals, fds_residuals = _get_residuals(dm_file_empty)

    assert fv_residuals.empty
    assert fds_residuals.empty

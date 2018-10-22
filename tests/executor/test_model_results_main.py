import pandas as pd
import pytest

from cascade.executor.model_results_main import _get_model_results


def test_get_model_results_inputs_ok(ihme):
    """at_mvid=265844 has 5850 rows, ode_mvid=102680 has 238836 rows
    """
    results_columns = ['model_version_id', 'year_id', 'location_id', 'sex_id',
                       'age_group_id', 'measure_id', 'mean', 'upper', 'lower']

    ode_model_version_id = 102680
    ode_model_type = "ODE"
    ode_results = _get_model_results(ode_model_version_id, ode_model_type)

    assert set(ode_results.columns) == set(results_columns)

    at_model_version_id = 265844
    at_model_type = "AT"
    at_results = _get_model_results(at_model_version_id, at_model_type)

    assert set(at_results.columns) == set(results_columns)

    at_row_index_8 = pd.Series([265844, 1990, 90, 1, 2, 16, 0.161961, 0.161961, 0.161961])
    at_row_index_8.index = at_results.columns

    pd.testing.assert_series_equal(at_results.iloc[8], at_row_index_8,
                                   check_exact=False, check_names=False)


def test_get_model_results_bad_model_type():
    """Expect an exception if model_type is not AT or ODE"""
    with pytest.raises(ValueError):
        model_version_id = 265844
        model_type = "NOT_AT_OR_ODE"
        _get_model_results(model_version_id, model_type)


def test_get_model_results_bad_model_version_id(ihme):
    """Expect an empty dataframe if model_version_id is not in the database"""

    at_model_version_id = 1
    at_model_type = "AT"
    at_results = _get_model_results(at_model_version_id, at_model_type)

    assert at_results.empty

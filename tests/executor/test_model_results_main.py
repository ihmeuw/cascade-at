import pandas as pd
import pytest

from cascade.executor.model_results_main import _get_model_results


def test_get_model_results_inputs_ok(ihme):
    """at_mvid=265844 has 5850 rows, ode_mvid=102680 has 238836 rows
    """
    results_columns = ['model_version_id', 'year_id', 'location_id', 'sex_id',
                       'age_group_id', 'measure_id', 'mean', 'upper', 'lower']

    ode_model_version_id = 102680
    db = "epi-prod"
    table = "fit"
    ode_results = _get_model_results(ode_model_version_id, db, table)

    assert set(ode_results.columns) == set(results_columns)

    at_model_version_id = 265844
    db = "at-dev"
    table = "fit"
    at_results = _get_model_results(at_model_version_id, db, table)

    assert set(at_results.columns) == set(results_columns)

    at_row_index_8 = pd.Series([265844, 1990, 90, 1, 2, 16, 0.161961, 0.161961, 0.161961])
    at_row_index_8.index = at_results.columns

    pd.testing.assert_series_equal(at_results.iloc[8], at_row_index_8,
                                   check_exact=False, check_names=False)


def test_get_model_results_bad_database_name(ihme):
    """Expect an exception if db not 'epi-dev', 'epi-prod', 'at-dev', or 'at-prod'"""
    with pytest.raises(SystemExit):
        model_version_id = 265844
        db = "not-at-dev"
        table = "fit"
        _get_model_results(model_version_id, db, table)


def test_get_model_results_bad_table_name(ihme):
    """Expect an exception if table not 'fit', 'final'"""
    with pytest.raises(SystemExit):
        model_version_id = 265844
        db = "at-dev"
        table = "not-fit"
        _get_model_results(model_version_id, db, table)


def test_get_model_results_bad_model_version_id(ihme):
    """Expect an empty dataframe if model_version_id is not in the database"""

    at_model_version_id = 1
    db = "at-dev"
    table = "fit"

    at_results = _get_model_results(at_model_version_id, db, table)

    assert at_results.empty

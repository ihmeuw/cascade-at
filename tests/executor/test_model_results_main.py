import pandas as pd
import pytest

from cascade.executor.model_results_main import _get_model_results


def test_get_model_results_inputs_ok(ihme):
    """at_mvid=265844 has 5850 rows, ode_mvid=102680 has 238836 rows
    """
    results_columns = ['model_version_id', 'year_id', 'location_id', 'sex_id',
                       'age_group_id', 'measure_id', 'mean', 'upper', 'lower']

    ode_model_version_id = 102680
    db = "epi"
    table = "fit"
    ode_results = _get_model_results(ode_model_version_id, db, table)

    assert set(ode_results.columns) == set(results_columns)

    at_model_version_id = 265844
    db = "dismod-at-dev"
    table = "fit"
    at_results = _get_model_results(at_model_version_id, db, table)

    assert set(at_results.columns) == set(results_columns)

    at_row_index_8 = pd.Series([265844, 1990, 90, 1, 2, 16, 0.161961, 0.161961, 0.161961])
    at_row_index_8.index = at_results.columns

    pd.testing.assert_series_equal(at_results.iloc[8], at_row_index_8,
                                   check_exact=False, check_names=False)


def test_get_model_results_bad_model_version_id_for_db_and_table(ihme):
    """Expect an exception if the mvid is not found in the specified db and table"""

    with pytest.raises(ValueError):
        model_version_id = 1
        db = "dismod-at-dev"
        table = "fit"
        _get_model_results(model_version_id, db, table)


def test_get_model_results_bad_model_version_id_all_locations(ihme):
    """Expect an exception if the mvid is not found in any of the locations"""

    with pytest.raises(ValueError):
        model_version_id = 1
        db = None
        table = None
        _get_model_results(model_version_id, db, table)


def test_get_model_results__multiple_finds(ihme):
    """Expect an exception if no db and table are given and the mvid is found in multiple locations"""

    with pytest.raises(ValueError):
        model_version_id = 265844
        db = None
        table = None
        _get_model_results(model_version_id, db, table)

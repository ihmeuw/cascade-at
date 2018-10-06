import pandas as pd
import pytest

from cascade.dismod.db.wrapper import DismodFile
from cascade.core.context import ExecutionContext
import cascade.saver.db.save_model_results_to_db
from cascade.saver.db.save_model_results_to_db import (
    _normalize_draws_df,
    _write_temp_draws_file_and_upload_model_results,
)


@pytest.fixture(scope="module")
def execution_context():
    defaults = {
        "modelable_entity_id": 1175,
        "model_version_id": None,
        "model_title": "GBD 2010 Best (dm-39976)",
        "db_env": "dev",
        "gbd_round_id": 5,
        "database": "dismod-at-dev",
        "year_id": 2010,
    }
    execution_context = ExecutionContext()
    execution_context.dismodfile = DismodFile()
    execution_context.dismodfile.node = pd.DataFrame(
        [{"node_id": 0, "c_location_id": 102, "node_name": "102", "parent": None}]
    )
    execution_context.parameters = defaults

    return execution_context


@pytest.fixture(scope="module")
def pre_normalized_draws_df():
    pre_normalized_draws_df = pd.DataFrame()
    pre_normalized_draws_df["age_upper"] = [65.0, 65.0, 65.0, 65.0, 75.0, 75.0, 75.0, 75]
    pre_normalized_draws_df["age_lower"] = [60.0, 60.0, 60.0, 60.0, 70.0, 70.0, 70.0, 70]
    pre_normalized_draws_df["node_id"] = [0, 0, 0, 0, 0, 0, 0, 0]
    pre_normalized_draws_df["integrand_id"] = [2, 2, 7, 7, 2, 2, 7, 7]
    pre_normalized_draws_df["weight_id"] = [0, 0, 0, 0, 0, 0, 0, 0]
    pre_normalized_draws_df["x_sex"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    pre_normalized_draws_df["time_lower"] = [1990.0, 1990.0, 1995.0, 1995.0, 2000.0, 2005.0, 2010.0, 2017.0]
    pre_normalized_draws_df["time_upper"] = [1990.0, 1990.0, 1995.0, 1995.0, 2000.0, 2005.0, 2010.0, 2017.0]
    pre_normalized_draws_df["draw_0"] = [1, 2, 3, 4, 1.1, 2.1, 3.1, 4.1]
    pre_normalized_draws_df["draw_1"] = [5, 6, 7, 8, 5.1, 6.1, 7.1, 8.1]
    pre_normalized_draws_df["draw_2"] = [9, 10, 11, 12, 9.1, 10.1, 11.1, 12.1]
    pre_normalized_draws_df["draw_3"] = [13, 14, 15, 16, 13.1, 14.1, 15.1, 16.1]

    return pre_normalized_draws_df


@pytest.fixture(scope="module")
def draws_df():
    draws_df = pd.DataFrame()
    draws_df["age_group_id"] = [17, 17, 17, 17, 19, 19, 19, 19]
    draws_df["location_id"] = [102, 102, 102, 102, 102, 102, 102, 102]
    draws_df["measure_id"] = [9, 9, 5, 5, 9, 9, 5, 5]
    draws_df["sex_id"] = [1, 1, 1, 1, 1, 1, 1, 1]
    draws_df["year_id"] = [1990, 1990, 1995, 1995, 2000, 2005, 2010, 2017]
    draws_df["draw_0"] = [1, 2, 3, 4, 1.1, 2.1, 3.1, 4.1]
    draws_df["draw_1"] = [5, 6, 7, 8, 5.1, 6.1, 7.1, 8.1]
    draws_df["draw_2"] = [9, 10, 11, 12, 9.1, 10.1, 11.1, 12.1]
    draws_df["draw_3"] = [13, 14, 15, 16, 13.1, 14.1, 15.1, 16.1]

    return draws_df


def save_results_fake(input_dir, input_file_pattern, modelable_entity_id, model_title, measures_to_save, **kwargs):

    mvid_df = pd.DataFrame()
    mvid_df["mvid"] = ["1234"]

    return mvid_df


def to_hdf_fake(file_path, key, *pargs, **kwargs):
    pass


@pytest.fixture
def fake_save_results_at(monkeypatch):
    monkeypatch.setattr(cascade.saver.db.save_model_results_to_db, "save_results_at", save_results_fake)


@pytest.fixture
def fake_write_hdf(monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_hdf", to_hdf_fake)


@pytest.fixture
def mock_db_queries(mocker):
    mock = mocker.patch("cascade.input_data.db.demographics.db_queries")
    mock.get_demographics.return_value = {
        "location_id": [210, 211, 212, 213, 214, 215, 216, 217, 218],
        "sex_id": [1, 2],
        "age_group_id": [9, 10, 11, 12, 13, 14, 15, 16, 17, 22],
        "year_id": [1990, 1995, 2000, 2005, 2010, 2017],
    }
    mock.get_age_metadata.return_value = pd.DataFrame(
        {
            "age_group_id": {9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 22: 235},
            "age_group_years_start": {
                9: 30.0,
                10: 35.0,
                11: 40.0,
                12: 45.0,
                13: 50.0,
                14: 55.0,
                15: 60.0,
                16: 65.0,
                17: 70.0,
                22: 95.0,
            },
            "age_group_years_end": {
                9: 35.0,
                10: 40.0,
                11: 45.0,
                12: 50.0,
                13: 55.0,
                14: 60.0,
                15: 65.0,
                16: 70.0,
                17: 75.0,
                22: 125.0,
            },
            "age_group_weight_value": {
                9: 0.0733159,
                10: 0.0677595,
                11: 0.0608903,
                12: 0.0546581,
                13: 0.0487362,
                14: 0.0425164,
                15: 0.0359617,
                16: 0.029149,
                17: 0.0213028,
                22: 0.000575352,
            },
        }
    )

    return mock


def test_normalize_draws_df(pre_normalized_draws_df, draws_df, execution_context, mock_db_queries):
    normalized_draws = _normalize_draws_df(pre_normalized_draws_df, execution_context)

    # ignoring the order of the rows and columns

    pd.testing.assert_frame_equal(draws_df, normalized_draws, check_like=True)


def test_write_temp_draws_file_and_upload_model_results_no_hdf_no_sr_call(
    draws_df, execution_context, fake_save_results_at, fake_write_hdf
):

    model_version_id = _write_temp_draws_file_and_upload_model_results(draws_df, execution_context)

    assert model_version_id == 1234


def test_write_temp_draws_file_and_upload_model_results_no_sr_call(draws_df, execution_context, fake_save_results_at):

    model_version_id = _write_temp_draws_file_and_upload_model_results(draws_df, execution_context)

    assert model_version_id == 1234

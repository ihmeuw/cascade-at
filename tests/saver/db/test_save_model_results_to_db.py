import pandas as pd
import pytest

from cascade.core.context import ExecutionContext
import cascade.saver.db.save_model_results_to_db
from cascade.saver.db.save_model_results_to_db import (
    write_temp_draws_file_and_upload_model_results)


@pytest.fixture(scope="module")
def execution_context():
    defaults = {"modelable_entity_id": 1175,
                "model_version_id": None,
                "model_title": "GBD 2010 Best (dm-39976)",
                "db_env": "dev",
                "gbd_round_id": 5,
                "year_id": 2010}
    execution_context = ExecutionContext()
    execution_context.parameters = defaults

    return execution_context


@pytest.fixture(scope="module")
def draws_df():
    draws_df = pd.DataFrame()
    draws_df["age_group_id"] = [7, 7, 7, 7, 20, 20, 20, 20]
    draws_df["location_id"] = [102, 102, 102, 102, 102, 102, 102, 102]
    draws_df["measure_id"] = [9, 9, 5, 5, 9, 9, 5, 5]
    draws_df["sex_id"] = [1, 2, 1, 2, 1, 2, 1, 2]
    draws_df["year_id"] = [
        1990, 1990, 1995, 1995,
        2000, 2005, 2010, 2017]
    draws_df["draw_0"] = [
        1, 2, 3, 4,
        1.1, 2.1, 3.1, 4.1]
    draws_df["draw_1"] = [
        5, 6, 7, 8,
        5.1, 6.1, 7.1, 8.1]
    draws_df["draw_2"] = [
        9, 10, 11, 12,
        9.1, 10.1, 11.1, 12.1]
    draws_df["draw_3"] = [
        13, 14, 15, 16,
        13.1, 14.1, 15.1, 16.1]

    return draws_df


def save_results_fake(
        input_dir,
        input_file_pattern,
        modelable_entity_id,
        model_title,
        measures_to_save,
        **kwargs):

    mvid_df = pd.DataFrame()
    mvid_df["mvid"] = ["1234"]

    return mvid_df


def to_hdf_fake(file_path, key, *pargs, **kwargs):
    pass


@pytest.fixture
def fake_save_results_at(monkeypatch):
    monkeypatch.setattr(cascade.saver.db.save_model_results_to_db, "save_results_at",
                        save_results_fake)


@pytest.fixture
def fake_write_hdf(monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_hdf",
                        to_hdf_fake)


def test_write_temp_draws_file_and_upload_model_results_no_hdf_no_sr_call(
        draws_df, execution_context, fake_save_results_at, fake_write_hdf):

    model_version_id = write_temp_draws_file_and_upload_model_results(
        draws_df, execution_context)

    assert model_version_id == 1234


def test_write_temp_draws_file_and_upload_model_results_no_sr_call(
        draws_df, execution_context, fake_save_results_at):

    model_version_id = write_temp_draws_file_and_upload_model_results(
        draws_df, execution_context)

    assert model_version_id == 1234

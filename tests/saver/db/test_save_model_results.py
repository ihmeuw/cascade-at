import pandas as pd
import pytest

try:
    from save_results._save_results import save_results_at
except ImportError:

    class DummySaveResults:
        def __getattr__(self, name):
            raise ImportError(f"Required package save_results not found")

    save_results_at = DummySaveResults()

from cascade.core.context import ExecutionContext
import cascade.saver
import cascade.saver.db.save_model_results as smr
from cascade.dismod.db.wrapper import _get_engine, DismodFile

DRAWS_INPUT_FILE_PATTERN = "all_draws.h5"


@pytest.fixture(scope="module")
def predict_df():
    predict_df = pd.DataFrame()
    predict_df["sample_index"] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    predict_df["avgint_id"] = [
        3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    predict_df["avg_integrand"] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

    return predict_df


@pytest.fixture(scope="module")
def avgint_df():
    avgint_df = pd.DataFrame()
    avgint_df["age_group_id"] = [7, 8, 9, 10, 11, 12,
                                 7, 8, 9, 10, 11, 12]
    avgint_df["avgint_id"] = [0, 1, 2, 3, 4,
                              5, 6, 7, 8, 9, 10, 11]
    avgint_df["integrand_id"] = [2, 2, 2, 2, 2, 2,
                                 7, 7, 7, 7, 7, 7]
    avgint_df["location_id"] = [139, 139, 139, 139, 139, 139,
                                139, 139, 139, 139, 139, 139]
    avgint_df["sex_id"] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    avgint_df["year_id"] = [1990, 1995, 2000, 2005, 2010, 2017,
                            1990, 1995, 2000, 2005, 2010, 2017]

    return avgint_df


@pytest.fixture(scope="module")
def dismod_file(avgint_df, predict_df):
    engine = _get_engine(None)
    dismod_file = DismodFile(engine, {}, {})
    dismod_file.avgint = avgint_df
    dismod_file.predict = predict_df

    return dismod_file


@pytest.fixture(scope="module")
def execution_context(dismod_file):
    defaults = {"modelable_entity_id": 1175,
                "model_version_id": None,
                "model_title": "GBD 2010 Best (dm-39976)",
                "db_env": "dev",
                "gbd_round_id": 5,
                "year_id": 2010}
    execution_context = ExecutionContext()
    execution_context.parameters = defaults
    execution_context._dismodfile = dismod_file

    return execution_context


@pytest.fixture(scope="module")
def execution_context_no_dismodfile():
    defaults = {"modelable_entity_id": 19777,
                "model_version_id": None,
                "model_title": None,
                "db_env": "dev"}
    execution_context_no_dismodfile = ExecutionContext()
    execution_context_no_dismodfile.parameters = defaults
    execution_context_no_dismodfile._dismodfile = None

    return execution_context_no_dismodfile


@pytest.fixture(scope="module")
def execution_context_no_meid(dismod_file):
    defaults = {"modelable_entity_id": None,
                "model_version_id": None,
                "model_title": None,
                "db_env": "dev"}
    execution_context_no_meid = ExecutionContext()
    execution_context_no_meid.parameters = defaults
    execution_context_no_meid._dismodfile = dismod_file

    return execution_context_no_meid


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


def get_draws_table(execution_context):
    return draws_df()


def save_results_fake(
        input_dir,
        input_file_pattern,
        modelable_entity_id,
        model_title,
        measures_to_save,
        model_version_id,
        **kwargs):

    mvid_df = pd.DataFrame()
    mvid_df["mvid"] = ["1234"]

    return mvid_df


def to_hdf_fake(file_path, key, *pargs, **kwargs):
    pass


@pytest.fixture
def fake_generate_draws(monkeypatch):
    monkeypatch.setattr(cascade.saver.db.save_model_results,
                        "generate_draws_table", get_draws_table)


@pytest.fixture
def fake_save_results_at(monkeypatch):
    monkeypatch.setattr(save_results._save_results, "save_results_at",
                        save_results_fake)


@pytest.fixture
def fake_write_hdf(monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_hdf",
                        to_hdf_fake)


def test_save_model_results_without_sr_call(
        fake_generate_draws, fake_save_results_at, execution_context):
    mvid = smr.save_model_results(execution_context)

    assert mvid == "1234"


def test_save_model_results_no_hdf_no_sr_call(
        fake_generate_draws, fake_save_results_at, fake_write_hdf,
        execution_context):
    mvid = smr.save_model_results(execution_context)

    assert mvid == "1234"


def test_save_model_results_no_dismodfile(fake_generate_draws,
                                          execution_context_no_dismodfile):
    with pytest.raises(ValueError):
        smr.save_model_results(execution_context_no_dismodfile)


def test_save_model_results_no_meid(fake_generate_draws,
                                    execution_context_no_meid):
    with pytest.raises(ValueError):
        smr.save_model_results(execution_context_no_meid)


@pytest.mark.skip
def test_save_model_results_real(execution_context):
    smr.save_model_results(execution_context)

import pandas as pd
import pytest

from cascade.core.context import ExecutionContext
from cascade.saver.save_model_results import save_model_results
from cascade.dismod.db.wrapper import _get_engine, DismodFile


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
                "model_version_id": 265085,
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


def test_save_model_results_no_dismodfile(execution_context_no_dismodfile):
    with pytest.raises(ValueError):
        save_model_results(execution_context_no_dismodfile)


@pytest.mark.skip
def test_save_model_results_real(ihme, execution_context):
    save_model_results(execution_context)

import pandas as pd
import pytest

from cascade.core.context import ExecutionContext
import cascade.saver
from cascade.saver.generate_draws import (
    generate_draws_table,
    pure_generate_draws,
    retrieve_prediction_tables)
from cascade.dismod.db.wrapper import _get_engine, DismodFile


@pytest.fixture(scope="module")
def avgint_df():
    avgint_df = pd.DataFrame()
    avgint_df["age_group_id"] = [7, 7, 7, 7]
    avgint_df["avgint_id"] = [0, 1, 2, 3]
    avgint_df["integrand_id"] = [2, 2, 7, 7]
    avgint_df["location_id"] = [102, 102, 102, 102]
    avgint_df["sex_id"] = [1, 2, 1, 2]
    avgint_df["year_id"] = [1990, 1990, 1990, 1990]

    return avgint_df


@pytest.fixture(scope="module")
def predict_df():
    predict_df = pd.DataFrame()
    predict_df["sample_index"] = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3]
    predict_df["avgint_id"] = [
        3, 0, 1, 2,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3]
    predict_df["avg_integrand"] = [
        4, 1, 2, 3,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16]

    return predict_df


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
                "model_version_id": 1,
                "model_title": "Cascade Model",
                "db_env": "dev"}
    execution_context = ExecutionContext()
    execution_context.parameters = defaults
    execution_context._dismodfile = dismod_file

    return execution_context


@pytest.fixture(scope="module")
def predict_df_ragged(predict_df):
    predict_df_copy = predict_df.copy()
    extra_data = {"sample_index": [4, 4], "avgint_id": [0, 1],
                  "avg_integrand": [17, 18]}
    extra_rows_df = pd.DataFrame(extra_data)

    predict_df_ragged = predict_df_copy.append(extra_rows_df)

    return predict_df_ragged


@pytest.fixture(scope="module")
def avgint_df_empty():
    avgint_df_empty = pd.DataFrame()

    return avgint_df_empty


@pytest.fixture(scope="module")
def expected_draws_df():
    expected_draws_df = pd.DataFrame()
    expected_draws_df["age_group_id"] = [7, 7, 7, 7]
    expected_draws_df["location_id"] = [102, 102, 102, 102]
    expected_draws_df["measure_id"] = [9, 9, 5, 5]
    expected_draws_df["sex_id"] = [1, 2, 1, 2]
    expected_draws_df["year_id"] = [1990, 1990, 1990, 1990]
    expected_draws_df["draw_0"] = [1, 2, 3, 4]
    expected_draws_df["draw_1"] = [5, 6, 7, 8]
    expected_draws_df["draw_2"] = [9, 10, 11, 12]
    expected_draws_df["draw_3"] = [13, 14, 15, 16]

    return expected_draws_df


def get_avgint_predict(execution_context):
    return avgint_df(), predict_df()


def get_draws(avgint, predict):
    return expected_draws_df()


@pytest.fixture
def fake_generate_draws(monkeypatch):
    monkeypatch.setattr(cascade.saver.generate_draws,
                        "retrieve_prediction_tables", get_avgint_predict)
    monkeypatch.setattr(cascade.saver.generate_draws,
                        "pure_generate_draws", get_draws)


def test_retrieve_prediction_tables(dismod_file):
    avgint, predict = retrieve_prediction_tables(dismod_file)

    pd.testing.assert_frame_equal(avgint, avgint_df(),
                                  check_like=True)

    pd.testing.assert_frame_equal(predict, predict_df(),
                                  check_like=True)


def test_generate_draws_table(fake_generate_draws, execution_context):

    draws_df = generate_draws_table(execution_context)

    pd.testing.assert_frame_equal(draws_df, expected_draws_df(),
                                  check_like=True)


def test_pure_generate_draws(avgint_df, predict_df, expected_draws_df):

    draws_df = pure_generate_draws(avgint_df, predict_df)

    assert draws_df.shape == (4, 9)

    expected_columns = ["location_id", "age_group_id", "year_id", "sex_id",
                        "measure_id", "draw_0", "draw_1", "draw_2", "draw_3"]

    assert set(draws_df.columns) == set(expected_columns)

    # ignoring the order of the rows and columns
    pd.testing.assert_frame_equal(draws_df, expected_draws_df, check_like=True)


def test_pure_generate_draws_mismatch_rows(avgint_df, predict_df_ragged):
    with pytest.raises(ValueError):
        pure_generate_draws(avgint_df, predict_df_ragged)


def test_pure_generate_draws_empty_avgint(avgint_df_empty, predict_df):
    with pytest.raises(ValueError):
        pure_generate_draws(avgint_df_empty, predict_df)

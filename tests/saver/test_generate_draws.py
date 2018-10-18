import pandas as pd
import pytest

from cascade.saver.generate_draws import generate_draws_table, pure_generate_draws
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
    predict_df["sample_index"] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    predict_df["avgint_id"] = [3, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    predict_df["avg_integrand"] = [4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    return predict_df


@pytest.fixture(scope="module")
def dismod_file(avgint_df, predict_df):
    engine = _get_engine(None)
    dismod_file = DismodFile(engine)
    dismod_file.avgint = avgint_df
    dismod_file.predict = predict_df

    return dismod_file


@pytest.fixture(scope="module")
def dismod_file_avgint_df_empty(predict_df):
    engine = _get_engine(None)
    dismod_file_avgint_df_empty = DismodFile(engine)
    dismod_file_avgint_df_empty.avgint = pd.DataFrame()
    dismod_file_avgint_df_empty.predict = predict_df

    return dismod_file_avgint_df_empty


@pytest.fixture(scope="module")
def dismod_file_predict_df_empty(avgint_df):
    engine = _get_engine(None)
    dismod_file_predict_df_empty = DismodFile(engine)
    dismod_file_predict_df_empty.avgint = avgint_df
    dismod_file_predict_df_empty.predict = pd.DataFrame()

    return dismod_file_predict_df_empty


@pytest.fixture(scope="module")
def predict_df_ragged(predict_df):
    predict_df_copy = predict_df.copy()
    extra_data = {"sample_index": [4, 4], "avgint_id": [0, 1], "avg_integrand": [17, 18]}
    extra_rows_df = pd.DataFrame(extra_data)

    predict_df_ragged = predict_df_copy.append(extra_rows_df)

    return predict_df_ragged


@pytest.fixture(scope="module")
def expected_draws_df():
    draws_df = pd.DataFrame()
    draws_df["age_group_id"] = [7, 7, 7, 7]
    draws_df["location_id"] = [102, 102, 102, 102]
    draws_df["integrand_id"] = [2, 2, 7, 7]
    draws_df["sex_id"] = [1, 2, 1, 2]
    draws_df["year_id"] = [1990, 1990, 1990, 1990]
    draws_df["draw_0"] = [1, 2, 3, 4]
    draws_df["draw_1"] = [5, 6, 7, 8]
    draws_df["draw_2"] = [9, 10, 11, 12]
    draws_df["draw_3"] = [13, 14, 15, 16]

    return draws_df


def test_generate_draws_table(dismod_file, expected_draws_df):

    draws_df = generate_draws_table(dismod_file)

    pd.testing.assert_frame_equal(draws_df, expected_draws_df, check_like=True)


def test_generate_draws_empty_avgint(dismod_file_avgint_df_empty):
    with pytest.raises(ValueError):
        generate_draws_table(dismod_file_avgint_df_empty)


def test_generate_draws_empty_predict(dismod_file_predict_df_empty):
    with pytest.raises(ValueError):
        generate_draws_table(dismod_file_predict_df_empty)


def test_pure_generate_draws(avgint_df, predict_df, expected_draws_df):

    draws_df = pure_generate_draws(avgint_df, predict_df)

    assert draws_df.shape == (4, 9)

    expected_columns = [
        "location_id",
        "age_group_id",
        "year_id",
        "sex_id",
        "integrand_id",
        "draw_0",
        "draw_1",
        "draw_2",
        "draw_3",
    ]

    assert set(draws_df.columns) == set(expected_columns)

    # ignoring the order of the rows and columns
    pd.testing.assert_frame_equal(draws_df, expected_draws_df, check_like=True)


def test_pure_generate_draws_mismatch_rows(avgint_df, predict_df_ragged):
    with pytest.raises(ValueError):
        pure_generate_draws(avgint_df, predict_df_ragged)

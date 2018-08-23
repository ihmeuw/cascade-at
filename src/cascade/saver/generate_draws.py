"""
Generate draws by getting data from the predict table and adding it to data
from the avgint table.
"""

import logging

import pandas as pd

CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)

INTEGRAND_ID_TO_MEASURE_ID_DF = pd.DataFrame([
    [0, 41],
    [1, 7],
    [2, 9],
    [3, 16],
    [4, 13],
    [5, 39],
    [6, 40],
    [7, 5],
    [8, 6],
    [9, 15],
    [10, 14],
    [11, 12],
    [12, 11]], columns=["integrand_id", "measure_id"])


def generate_draws_table(dm_file):

    avgint_df = dm_file.avgint
    predict_df = dm_file.predict

    if avgint_df.empty:
        raise ValueError("avgint_df has no rows")

    if predict_df.empty:
        raise ValueError("predict_df has no rows")

    draws_df = pure_generate_draws(avgint_df, predict_df)

    return draws_df


def pure_generate_draws(avgint_df, predict_df):
    """
    Generates a draws df which has a column for each draw, and id columns
    for location, age_group, year, sex, and measure.  If the predict table
    has 10 times as many rows as the avgint table, 10 draw columns will be
    generated, and the draws df will have 10 + 5 cols and the same number of
    rows as the avgint table.
    """

    avgint_rows = len(avgint_df)
    predict_rows = len(predict_df)

    num_draws = predict_df["sample_index"].nunique()

    if predict_rows % avgint_rows != 0:
        raise ValueError(
            "Predict table does not have an integer number of "
            "predictions of the avgint table")

    avgint = avgint_df.merge(
        INTEGRAND_ID_TO_MEASURE_ID_DF, how="left", on="integrand_id")

    draws = predict_df.pivot(index="avgint_id", columns="sample_index",
                             values="avg_integrand")

    draws.columns = ["draw_" + str(i) for i in range(num_draws)]

    draws = draws.reset_index(level=["avgint_id"])

    draw_df = avgint.merge(draws, how="left", on="avgint_id").drop(
        columns=["avgint_id", "integrand_id"])

    return draw_df

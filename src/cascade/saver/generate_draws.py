"""
Generate draws by getting data from the predict table and adding it to data
from the avgint table.
"""

import logging

from cascade.saver import INTEGRAND_ID_TO_MEASURE_ID_DF


CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)


def retrieve_prediction_tables(dm_file):
    """The avgint and predict tables are needed to create a draws df."""

    # If these tables are not in the db, an exception will be thrown
    CODELOG.debug("Retrieving avgint and predict tables")

    avgint_df = dm_file.avgint

    predict_df = dm_file.predict

    CODELOG.debug("avgint has {} rows".format(avgint_df.shape[0]))
    CODELOG.debug("predict has {} rows".format(predict_df.shape[0]))

    MATHLOG.info("avgint has {} rows".format(avgint_df.shape[0]))
    MATHLOG.info("predict has {} rows".format(predict_df.shape[0]))

    if avgint_df.empty:
        raise ValueError("avgint_df has no rows")

    if predict_df.empty:
        raise ValueError("predict_df has no rows")

    return avgint_df, predict_df


def generate_draws_table(dm_file):

    avgint_df, predict_df = retrieve_prediction_tables(dm_file)

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

    avgint_rows = avgint_df.shape[0]
    predict_rows = predict_df.shape[0]

    if not avgint_rows:
        raise ValueError("No rows in avgint_df")

    num_draws = predict_rows // avgint_rows

    CODELOG.debug(f"number of draws: {num_draws}")

    if predict_rows % avgint_rows:
        raise ValueError(
            "Predict table does not have an integer number of "
            "predictions of the avgint table")

    CODELOG.debug(f"avgint shape (expect 6 cols): {avgint_df.shape}")
    CODELOG.debug(f"avgint row 1: {avgint_df.iloc[0]}")

    avgint = avgint_df.merge(
        INTEGRAND_ID_TO_MEASURE_ID_DF, how="left", on="integrand_id")

    CODELOG.debug(f"avgint shape (expect 7 cols): {avgint.shape}")

    CODELOG.debug(f"predict_df shape (expect 3 cols): {predict_df.shape}")

    CODELOG.debug(f"Pivoting the predict_df")

    draws = predict_df.pivot(index="avgint_id", columns="sample_index",
                             values="avg_integrand")

    draws.columns = ["draw_" + str(i) for i in range(num_draws)]

    draws.reset_index(level=["avgint_id"], inplace=True)

    CODELOG.debug(f"draws shape: {draws.shape}")
    CODELOG.debug(f"draws cols: {draws.columns}")

    draw_df = avgint.merge(draws, how="left", on="avgint_id").drop(
        columns=["avgint_id", "integrand_id"])

    CODELOG.debug(f"draw_df shape (expect {num_draws+5} cols):{draw_df.shape}")
    CODELOG.debug(f"draw_df row 1 {draw_df.iloc[0]}")

    if not draw_df.shape[0]:
        raise ValueError("Draws table does not have any rows.")

    return draw_df

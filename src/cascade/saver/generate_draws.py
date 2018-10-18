"""
Generate draws by getting data from the predict table and adding it to data
from the avgint table.
"""

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def generate_draws_table(dm_file):
    """
    Creates a draws table based on the avgint and predict tables.

    Args:
        dm_file (DismodFile): contains avgint and predict tables

    Returns:
        pd.DataFrame

    """

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

    num_draws = predict_df["sample_index"].fillna(-1).nunique()

    if predict_rows % avgint_rows != 0:
        raise ValueError("Predict table does not have an integer number of predictions of the avgint table")

    draws = predict_df.pivot(index="avgint_id", columns="sample_index", values="avg_integrand")

    draws.columns = ["draw_" + str(i) for i in range(num_draws)]

    draws = draws.reset_index(level=["avgint_id"])

    draws_df = avgint_df.merge(draws, how="left", on="avgint_id").drop(columns=["avgint_id"])

    return draws_df

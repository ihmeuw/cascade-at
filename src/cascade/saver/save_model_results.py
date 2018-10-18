"""
Saves results from a model fit to a database which the EpiViz can access.
"""
from cascade.saver.generate_draws import generate_draws_table
from cascade.saver.db.save_model_results_to_db import save_model_results_to_db

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def save_model_results(execution_context):
    """
    We want to visualize model results using EpiViz.  To do this, we need to
    generate draws from a model fit, and save those draws in a database
    that EpiViz can access.

    Args:
        execution_context (ExecutionContext): contains model id data

    Returns:
        (int) of the model_version_id returned by save_results
    """

    dm_file = execution_context.dismodfile

    if not dm_file:
        raise ValueError("Must provide a dismod file")

    draws_df = generate_draws_table(dm_file)

    model_version_id = save_model_results_to_db(draws_df, execution_context)

    return model_version_id

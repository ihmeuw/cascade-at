"""
Saves results from a model fit to a database which the EpiViz can access.
"""
import logging

from cascade.saver.generate_draws import generate_draws_table
from cascade.saver.db.save_model_results_to_db import write_temp_draws_file_and_upload_model_results

CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)


def save_model_results(execution_context):
    """
    We want to visualize model results using EpiViz.  To do this, we need to
    generate draws from a model fit, and save those draws in a database
    that EpiViz can access.

    Args:
        execution_context (ExecutionContext): contains model id data

    Returns:
        (int) of the mvid returned by save_results
    """

    dm_file = execution_context.dismodfile

    if not dm_file:
        raise ValueError("Must provide a dismod file")

    draws_df = generate_draws_table(dm_file)

    model_version_id = write_temp_draws_file_and_upload_model_results(
        draws_df, execution_context)

    return model_version_id

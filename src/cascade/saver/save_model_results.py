"""
Saves results from a model fit to a database which the EpiViz can access.
"""
import logging

import save_results._save_results as sr

from cascade.saver.generate_draws import generate_draws_table

from cascade.saver import DRAWS_INPUT_FILE_PATTERN, MODEL_TITLE


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
        (str) of the mvid returned by save_results

    """

    dm_file = execution_context.dismodfile

    if not dm_file:
        raise ValueError("Must provide a dismod file")

    draws_df = generate_draws_table(dm_file)

    CODELOG.debug(draws_df.columns)
    CODELOG.debug(draws_df.iloc[0])

    with execution_context.scratch_dir() as scratch_dir:
        CODELOG.debug(scratch_dir)
        file_path = (scratch_dir / DRAWS_INPUT_FILE_PATTERN)

        model_version_id = execution_context.parameters.model_version_id
        modelable_entity_id = execution_context.parameters.modelable_entity_id

        CODELOG.debug(f"modelable_entity_id {modelable_entity_id}")

        if not modelable_entity_id:
            raise ValueError("Must provide a modelable entity id")

        model_title = execution_context.parameters.model_title

        if not model_title:
            model_title = MODEL_TITLE

        measures_to_save = list(draws_df["measure_id"].unique())

        CODELOG.debug(f"Saving measures: {measures_to_save}")

        CODELOG.debug("Writing Draws File")

        draws_df.to_hdf(
            str(file_path),
            "draws",
            format="table",
            data_columns=[
                "age_group_id",
                "location_id",
                "measure_id",
                "sex_id",
                "year_id"])

        CODELOG.debug("Saving Results")

        db_env = execution_context.parameters.db_env

        model_version_id_df = sr.save_results_at(
            str(scratch_dir),
            DRAWS_INPUT_FILE_PATTERN,
            modelable_entity_id,
            model_title,
            measures_to_save,
            model_version_id,
            db_env=db_env)

        CODELOG.debug(f"mvid_df: {model_version_id_df.iloc[0, 0]}")

        # context.scratch_dir() handles clean-up

        return str(model_version_id_df.iloc[0, 0])

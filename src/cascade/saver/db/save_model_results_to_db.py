"""
Saves results from a model fit to a database which the EpiViz can access.
"""
import logging
from pathlib import Path
import tempfile

import pandas as pd

try:
    from save_results._save_results import save_results_at
except ImportError:

    class DummySaveResults:
        def __getattr__(self, name):
            raise ImportError(f"Required package save_results not found")

    save_results_at = DummySaveResults()

DRAWS_INPUT_FILE_PATTERN = "all_draws.h5"

INTEGRAND_ID_TO_MEASURE_ID_DF = pd.DataFrame(
    [
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
        [12, 11],
    ],
    columns=["integrand_id", "measure_id"],
)

CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)


def _normalize_draws_df(draws_df):
    """The database stores measure_id rather than integrand_id."""

    draws = draws_df.merge(INTEGRAND_ID_TO_MEASURE_ID_DF, how="left", on="integrand_id").drop(columns=["integrand_id"])

    return draws


def _write_temp_draws_file_and_upload_model_results(draws_df, execution_context):
    """

    Args:
        draws_df (pd.DataFrame): the draws data to upload
        execution_context (ExecutionContext): contains model id data

    Returns:
        (int) of the model_version_id returned by save_results

    """

    with tempfile.TemporaryDirectory() as tmpdirname:

        file_path = Path(tmpdirname) / DRAWS_INPUT_FILE_PATTERN

        draws_df.to_hdf(
            str(file_path),
            "draws",
            format="table",
            data_columns=["age_group_id", "location_id", "measure_id", "sex_id", "year_id"],
        )
        import pdb

        pdb.set_trace()

        CODELOG.debug("Saving Results to DB")

        modelable_entity_id = execution_context.parameters.modelable_entity_id
        model_title = execution_context.parameters.model_title
        measures_to_save = list(draws_df["measure_id"].unique())
        model_version_id = execution_context.parameters.model_version_id
        if "prod" in execution_context.parameters.database:
            db_env = "prod"
        else:
            db_env = "dev"

        model_version_id_df = save_results_at(
            tmpdirname,
            DRAWS_INPUT_FILE_PATTERN,
            modelable_entity_id,
            model_title,
            measures_to_save,
            model_version_id=model_version_id,
            db_env=db_env,
        )

        CODELOG.debug(f"model_version_id_df: {model_version_id_df.iloc[0, 0]}")

        return int(model_version_id_df.iloc[0, 0])


def save_model_results_to_db(draws_df, execution_context):
    """
    Gets the draws data normalized for the db and then uploads it.

    Args:
        draws_df (pd.DataFrame): the draws data to upload, after normalizing it
        execution_context (ExecutionContext): contains model id data

    Returns:
        (int) of the model_version_id returned by save_results

    """

    draws_df = _normalize_draws_df(draws_df)

    model_version_id = _write_temp_draws_file_and_upload_model_results(draws_df, execution_context)

    return model_version_id

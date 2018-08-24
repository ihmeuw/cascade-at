"""
Saves results from a model fit to a database which the EpiViz can access.
"""
import logging
from pathlib import Path
import tempfile

try:
    from save_results._save_results import save_results_at
except ImportError:

    class DummySaveResults:
        def __getattr__(self, name):
            raise ImportError(f"Required package save_results not found")

    save_results_at = DummySaveResults()

DRAWS_INPUT_FILE_PATTERN = "all_draws.h5"

CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)


def write_temp_draws_file_and_upload_model_results(draws_df, execution_context):
    """

    Args:
        draws_df (pd.DataFrame): the draws data to upload
        execution_context (ExecutionContext): contains model id data

    Returns:
        (int) of the mvid returned by save_results

    """

    with tempfile.TemporaryDirectory() as tmpdirname:

        file_path = (Path(tmpdirname) / DRAWS_INPUT_FILE_PATTERN)

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

        CODELOG.debug("Saving Results to DB")

        modelable_entity_id = execution_context.parameters.modelable_entity_id
        model_title = execution_context.parameters.model_title
        measures_to_save = list(draws_df["measure_id"].unique())
        model_version_id = execution_context.parameters.model_version_id
        db_env = execution_context.parameters.db_env

        model_version_id_df = save_results_at(
            tmpdirname,
            DRAWS_INPUT_FILE_PATTERN,
            modelable_entity_id,
            model_title,
            measures_to_save,
            model_version_id=model_version_id,
            db_env=db_env)

        CODELOG.debug(f"model_version_id_df: {model_version_id_df.iloc[0, 0]}")

        return int(model_version_id_df.iloc[0, 0])

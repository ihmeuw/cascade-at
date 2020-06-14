import logging
import sys

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, ParentLocationID, SexID, LogLevel
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.dismod_extractor import DismodExtractor
from cascade_at.saver.results_handler import ResultsHandler

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    ParentLocationID(),
    SexID(),
    LogLevel()
])


def format_upload(model_version_id: int, parent_location_id: int, sex_id: int) -> None:
    """
    Takes a dismod database that has had predict run on it and converts the predictions
    into the format needed for the IHME Epi Databases. Also uploads inputs to tier 3 which
    allows us to view those inputs in EpiViz.

    Parameters
    ----------
    model_version_id
        The model version ID to format results for
    parent_location_id
        The parent location ID to upload for
    sex_id
        The sex ID to upload results for
    """
    context = Context(model_version_id=model_version_id)
    inputs, alchemy, settings = context.read_inputs()

    if not inputs.csmr.raw.empty:
        LOG.info("Uploading CSMR to t3")
        inputs.csmr.attach_to_model_version_in_db(
            model_version_id=model_version_id,
            conn_def=context.model_connection
        )

    LOG.info("Extracting results from DisMod SQLite Database.")
    dismod_file = context.db_file(location_id=parent_location_id, sex_id=sex_id)
    da = DismodExtractor(path=dismod_file)
    predictions = da.format_predictions_for_ihme()

    LOG.info("Saving the results.")
    rh = ResultsHandler(model_version_id=model_version_id)
    rh.save_draw_files(df=predictions, directory=context.draw_dir)
    rh.upload_summaries(directory=context.draw_dir, conn_def=context.model_connection)


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    format_upload(
        model_version_id=args.model_version_id,
        parent_location_id=args.parent_location_id,
        sex_id=args.sex_id
    )

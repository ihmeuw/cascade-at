import logging
import sys

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, LogLevel, BoolArg
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.saver.results_handler import ResultsHandler

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    BoolArg('--final', help='whether or not to upload final results'),
    BoolArg('--fit', help='whether or not to upload model fits'),
    BoolArg('--prior', help='whether or not to upload model priors'),
    LogLevel()
])


def upload_prior(context: Context, rh: ResultsHandler) -> None:
    """
    Uploads the saved priors to the epi database in the table
    epi.model_prior..

    Parameters
    ----------
    rh
        a Results Handler object
    context
        A context object
    """
    rh.upload_summaries(
        directory=context.prior_dir,
        conn_def=context.model_connection,
        table='model_prior'
    )


def upload_fit(context: Context, rh: ResultsHandler) -> None:
    """
    Uploads the saved final results to a the epi database in the table
    epi.model_estimate_fit.
    .
    Parameters
    ----------
    rh
        a Results Handler object
    context
        A context object
    """
    rh.upload_summaries(
        directory=context.fit_dir,
        conn_def=context.model_connection,
        table='model_estimate_fit'
    )


def upload_final(context: Context, rh: ResultsHandler) -> None:
    """
    Uploads the saved final results to a the epi database in the table
    epi.model_estimate_final.

    Parameters
    ----------
    rh
        a Results Handler object
    context
        A context object
    """
    rh.upload_summaries(
        directory=context.draw_dir,
        conn_def=context.model_connection,
        table='model_estimate_final'
    )


def format_upload(model_version_id: int, final: bool = False, fit: bool = False,
                  prior: bool = False) -> None:

    context = Context(model_version_id=model_version_id)
    print (f'Saving to connection {context.model_connection}')
    rh = ResultsHandler()

    if final:
        upload_final(context=context, rh=rh)
    if fit:
        upload_fit(context=context, rh=rh)
    if prior:
        upload_prior(context=context, rh=rh)


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    format_upload(
        model_version_id=args.model_version_id,
        fit=args.fit,
        prior=args.prior,
        final=args.final
    )


if __name__ == '__main__':
    main()

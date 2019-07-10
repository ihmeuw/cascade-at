import faulthandler
import os
from bdb import BdbQuit
from pprint import pformat
from textwrap import fill

from cascade.core import getLoggers, __version__
from cascade.core.db import use_local_odbc_ini
from cascade.runner.argument_parser import DMArgumentParser
from cascade.runner.cascade_logging import logging_config
from cascade.input_data.configuration import SettingsError
from cascade.runner.graph_execute import run_job_graph

CODELOG, MATHLOG = getLoggers(__name__)


def main(app, args):
    # get or make settings
    if args.create_settings:
        app.create_settings(args)
    else:
        app.load_settings(args)

    # save settings if requested
    if args.save_settings:
        app.save_settings()

    if run:
        if args.grid_engine:
            backend = "grid_engine"
        else:
            backend = "single_process"
        work = app.sub_graph_to_run(args)
        continuation = False
        run_job_graph(work, backend, continuation)


def entry(app, args=None):
    """
    This is most of the main, but it needs to be initialized with
    an Application instance::

        if __name__ == "__main__":
            app = Application()
            entry(app)

    """
    readable_by_all = 0o0002
    os.umask(readable_by_all)
    faulthandler.enable()

    parser = make_parser()
    app.add_arguments(parser)
    args = parser.parse_args(args)
    logging_config(args)

    MATHLOG.debug(f"Cascade version {__version__}.")
    if "JOB_ID" in os.environ:
        MATHLOG.info(f"Job id is {os.environ['JOB_ID']} on cluster {os.environ.get('SGE_CLUSTER_NAME', '')}")

    try:
        use_local_odbc_ini()
        main(app, args)
    except SettingsError as e:
        MATHLOG.error(str(e))
        CODELOG.error(f"Form data:{os.linesep}{pformat(e.form_data)}")
        error_lines = list()
        for error_spot, human_spot, error_message in e.form_errors:
            if args.settings_file is not None:
                error_location = error_spot
            else:
                error_location = human_spot
            error_lines.append(f"\t{error_location}: {error_message}")
        MATHLOG.error(f"Form validation errors:{os.linesep}{os.linesep.join(error_lines)}")
        exit(1)
    except BdbQuit:
        pass
    except Exception:
        if args.pdb:
            import pdb
            import traceback

            traceback.print_exc()
            pdb.post_mortem()
        else:
            MATHLOG.exception(f"Uncaught exception in {os.path.basename(__file__)}")
            raise


def make_parser():
    parser = DMArgumentParser("Run DismodAT from Epiviz")
    run_parser = parser.add_argument_group(
        "runner",
        fill("These commands affect how the jobs run, whether they "
             "are run in this process or started by Grid Engine"),
    )
    run_parser.add_argument(
        "--num-processes", type=int, default=4,
        help="How many subprocesses to start if we start subproceses.",
    )
    run_parser.add_argument(
        "--pdb", action="store_true",
        help="Drops you into the debugger on exception"
    )
    run_parser.add_argument(
        "--single-use-machine", action="store_true",
        help="True if processes should use a high nice value."
    )
    run_parser.add_argument(
        "--grid-engine", action="store_true",
        help=fill(
            "Start Grid Engine jobs. If this isn't specified, then "
            "the application will run jobs in this process."
        ),
    )
    return parser

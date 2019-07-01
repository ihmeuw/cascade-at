"""
Entry point for running a the work of a single location in an EpiViz-AT cascade.
"""
import faulthandler
import os
from bdb import BdbQuit
from datetime import timedelta
from pathlib import Path
from pprint import pformat
from timeit import default_timer

from cascade.core import getLoggers, __version__
from cascade.core.db import use_local_odbc_ini
from cascade.executor.argument_parser import DMArgumentParser
from cascade.executor.cascade_logging import logging_config
from cascade.executor.execution_context import make_execution_context
from cascade.executor.job_definitions import job_graph_from_settings
from cascade.input_data.configuration import SettingsError
from cascade.input_data.db.configuration import load_settings
from cascade.input_data.db.locations import location_hierarchy
from cascade.runner.entry import run_job_graph

CODELOG, MATHLOG = getLoggers(__name__)


def generate_plan(execution_context, args):
    """Creates a plan for the whole hierarchy, of which this job will be one."""
    settings = load_settings(execution_context, args.meid, args.mvid, args.settings_file)
    locations = location_hierarchy(
        location_set_version_id=settings.location_set_version_id,
        gbd_round_id=settings.gbd_round_id
    )
    return job_graph_from_settings(locations, settings, args), settings


def configure_execution_context(execution_context, args, settings):
    if args.infrastructure:
        execution_context.parameters.organizational_mode = "infrastructure"
    else:
        execution_context.parameters.organizational_mode = "local"

    execution_context.parameters.base_directory = args.base_directory

    for param in ["modelable_entity_id", "model_version_id"]:
        setattr(execution_context.parameters, param, getattr(settings.model, param))


def subgraph_from_args(nodes, args):
    """Given all nodes in the global model, choose which to run
    in this execution."""
    for search in ["location_id", "recipe", "sex", "name"]:
        if search in args:
            nodes = [n for n in nodes if getattr(n, search) == getattr(args, search)]
    return nodes


def main(args):
    start_time = default_timer()
    execution_context = make_execution_context(gbd_round_id=6, num_processes=args.num_processes)
    job_graph, settings = generate_plan(execution_context, args)
    configure_execution_context(execution_context, args, settings)

    subgraph = subgraph_from_args(job_graph.nodes, args)
    if len(subgraph) == 0:
        MATHLOG.warning(f"There are no jobs selected for arguments {args}")
        raise RuntimeError(f"No nodes selected by {args}.")
    what_to_run = dict(
        job_graph=job_graph,
        subgraph=subgraph,
        execution_context=execution_context,
    )
    # This requests running one draw of the sample draws.
    if "task_index" in args and args.task_index is not None:
        what_to_run["task_index"] = args.task_index

    if args.single_process:
        backend = "single_process"
    else:
        backend = "grid_engine"
    continuation = False
    run_job_graph(what_to_run, backend, continuation)
    elapsed_time = timedelta(seconds=default_timer() - start_time)
    MATHLOG.debug(f"Completed successfully in {elapsed_time}")


def entry(args=None):
    """Allow passing args for testing."""
    readable_by_all = 0o0002
    os.umask(readable_by_all)
    faulthandler.enable()

    args = parse_arguments(args)
    logging_config(args)

    MATHLOG.debug(f"Cascade version {__version__}.")
    if "JOB_ID" in os.environ:
        MATHLOG.info(f"Job id is {os.environ['JOB_ID']} on cluster {os.environ.get('SGE_CLUSTER_NAME', '')}")

    try:
        if args.skip_cache:
            args.no_upload = True

        use_local_odbc_ini()
        main(args)
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


def parse_arguments(args):
    parser = DMArgumentParser("Run DismodAT from Epiviz")
    parser.add_argument("db_file_path", type=Path, default="z.db")
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--infrastructure", action="store_true",
                        help="Whether we are running as infrastructure component")
    parser.add_argument("--base-directory", type=Path, default=".",
                        help="Directory in which to find and store files.")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--db-only", action="store_true")
    parser.add_argument("-b", "--bundle-file", type=Path)
    parser.add_argument("-s", "--bundle-study-covariates-file", type=Path)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--num-processes", type=int, default=4,
                        help="How many subprocesses to start.")
    parser.add_argument("--num-samples", type=int, help="Override number of samples.")
    parser.add_argument("--pdb", action="store_true",
                        help="Drops you into the debugger on exception")
    parser.add_argument("--location-id", type=int, help="location ID for this work")
    parser.add_argument("--sex", type=str, help="sex as male, female, both")
    parser.add_argument("--recipe", type=str, help="name of the recipe")
    parser.add_argument("--name", type=str, help="job within the recipe")
    parser.add_argument("--task-index", type=int, help="index of draw")
    parser.add_argument("--single-process", action="store_true",
                        help="Run within this process, not in a subprocess.")
    return parser.parse_args(args)


if __name__ == "__main__":
    entry()

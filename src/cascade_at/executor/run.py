import subprocess
import sys
from typing import Optional

import logging

from cascade_at.cascade.cascade_commands import Drill, TraditionalCascade
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, BoolArg, LogLevel, NSim, StrArg
from cascade_at.inputs.locations import LocationDAG
from cascade_at.jobmon.workflow import jobmon_workflow_from_cascade_command
from cascade_at.settings.settings import settings_from_model_version_id

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    BoolArg('--jobmon', help='whether or not to use jobmon to run the cascade'
                             'or just run as a sequence of command line tasks'),
    BoolArg('--make', help='whether or not to make the file structure for the cascade'),
    NSim(),
    StrArg('--addl-workflow-args', help='additional info to append to workflow args, to re-do models',
           required=False),
    BoolArg('--skip-configure'),
    LogLevel()
])


def run(model_version_id: int, jobmon: bool = True, make: bool = True, n_sim: int = 10,
        addl_workflow_args: Optional[str] = None, skip_configure: bool = False) -> None:
    """
    Runs the whole cascade or drill for a model version (whichever one is specified
    in the model version settings).

    Creates a cascade command and a bunch of cascade operations based
    on the model version settings. More information on this structure
    is in :ref:`executor`.

    Parameters
    ----------
    model_version_id
        The model version to run
    jobmon
        Whether or not to use Jobmon. If not using Jobmon, executes
        the commands in sequence in this session.
    make
        Whether or not to make the directory structure for the databases, inputs, and outputs.
    n_sim
        Number of simulations to do going down the cascade
    addl_workflow_args
        Additional workflow args to add to the jobmon workflow name
        so that it is unique if you're testing
    skip_configure
        Skip configuring the inputs because
    """
    LOG.info(f"Starting model for {model_version_id}.")

    context = Context(
        model_version_id=model_version_id,
        make=make,
        configure_application=True
    )
    context.update_status(status='Submitted')

    settings = settings_from_model_version_id(
        model_version_id=model_version_id,
        conn_def=context.model_connection
    )
    dag = LocationDAG(location_set_version_id=settings.location_set_version_id,
                      gbd_round_id=settings.gbd_round_id)

    if settings.model.drill == 'drill':
        cascade_command = Drill(
            model_version_id=model_version_id,
            drill_parent_location_id=settings.model.drill_location_start,
            drill_sex=settings.model.drill_sex
        )
    elif settings.model.drill == 'cascade':

        location_start = None
        sex = None

        if isinstance(settings.model.drill_location_start, int):
            location_start = settings.model.drill_location_start
        if isinstance(settings.model.drill_sex, int):
            sex = settings.model.drill_sex

        cascade_command = TraditionalCascade(
            model_version_id=model_version_id,
            split_sex=settings.model.split_sex == 'most_detailed',
            dag=dag,
            n_sim=n_sim,
            location_start=settings.model.drill_location_start,
            sex=sex,
            skip_configure=skip_configure
        )
    else:
        raise NotImplementedError(f"The drill/cascade setting {settings.model.drill} is not implemented.")

    if jobmon:
        LOG.info("Configuring jobmon.")
        wf = jobmon_workflow_from_cascade_command(cc=cascade_command, context=context,
                                                  addl_workflow_args=addl_workflow_args)
        wf_run = wf.run(
            seconds_until_timeout=60*60*24*3,
            resume=True
        )
        if wf_run.status != 'D':
            context.update_status(status='Failed')
            raise RuntimeError("Jobmon workflow failed.")
    else:
        LOG.info("Running without jobmon.")
        for c in cascade_command.get_commands():
            LOG.info(f"Running {c}.")
            process = subprocess.run(
                c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if process.returncode:
                context.update_status(status='Failed')
                raise RuntimeError(f"Command {c} failed with error"
                                   f"{process.stderr.decode()}")

    context.update_status(status='Complete')


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    run(
        model_version_id=args.model_version_id,
        jobmon=args.jobmon,
        make=args.make,
        n_sim=args.n_sim,
        addl_workflow_args=args.addl_workflow_args,
        skip_configure=args.skip_configure
    )


if __name__ == '__main__':
    main()

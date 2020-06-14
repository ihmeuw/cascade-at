import logging
import subprocess
import sys

from cascade_at.cascade.cascade_commands import Drill
from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, BoolArg, LogLevel
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.jobmon.workflow import jobmon_workflow_from_cascade_command
from cascade_at.settings.settings import settings_from_model_version_id

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    BoolArg('--jobmon', help='whether or not to use jobmon to run the cascade'
                             'or just run as a sequence of command line tasks'),
    BoolArg('--make', help='whether or not to make the file structure for the cascade'),
    LogLevel()
])


def run(model_version_id: int, jobmon: bool = True, make: bool = True) -> None:
    """
    Runs the whole cascade or drill for a model version (which one is specified
    in the model version settings).

    Parameters
    ----------
    model_version_id
        The model version to run
    jobmon
        Whether or not to use Jobmon. If not using Jobmon, executes
        the commands in sequence in this session.
    make
        Whether or not to make the directory structure for the databases, inputs, and outputs.
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

    if settings.model.drill == 'drill':
        cascade_command = Drill(
            model_version_id=model_version_id,
            drill_parent_location_id=settings.model.drill_location_start,
            drill_sex=settings.model.drill_sex
        )
    elif settings.model.drill == 'cascade':
        raise NotImplementedError("Cascade is not implemented yet for Cascade-AT.")
    else:
        raise NotImplementedError(f"The drill/cascade setting {settings.model.drill} is not implemented.")

    if jobmon:
        LOG.info("Configuring jobmon.")
        wf = jobmon_workflow_from_cascade_command(cc=cascade_command, context=context)
        error = wf.run()
        if error:
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
        make=args.make
    )


if __name__ == '__main__':
    main()

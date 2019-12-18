import logging
import subprocess
from argparse import ArgumentParser

from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.cascade.cascade_commands import CASCADE_COMMANDS
from cascade_at.settings.settings import settings_from_model_version_id
from cascade_at.context.model_context import Context
from cascade_at.inputs.locations import LocationDAG
from cascade_at.jobmon.workflow import jobmon_workflow_from_cascade_command

LOG = get_loggers(__name__)


def get_args():
    """
    Parse the arguments for running a Cascade model!
    :return: parsed args
    """
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True)
    parser.add_argument("--jobmon", action='store_true',
                        help="whether or not to use jobmon to run the cascade or just"
                             "run as a sequence of command line tasks")
    parser.add_argument("--make", action='store_true',
                        help="whether or not to make the file structure for cascade")
    parser.add_argument("--loglevel", type=str, required=False, default="info")
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])
    LOG.info(f"Starting model for {args.model_version_id}.")

    context = Context(
        model_version_id=args.model_version_id,
        make=True,
        configure_application=True
    )
    context.update_status(status='Submitted')

    settings = settings_from_model_version_id(
        model_version_id=args.model_version_id,
        conn_def=context.model_connection
    )

    if settings.model.drill == 'drill':
        cascade_command = CASCADE_COMMANDS['drill'](
            model_version_id=args.model_version_id,
            drill_parent_location_id=settings.model.drill_location_start,
            drill_sex=settings.model.drill_sex
        )
    elif settings.model.drill == 'cascade':
        raise NotImplementedError("Cascade is not implemented yet for Cascade-AT.")
    else:
        raise NotImplementedError(f"The drill/cascade setting {settings.model.drill} is not implemented.")

    if args.jobmon:
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


if __name__ == '__main__':
    main()

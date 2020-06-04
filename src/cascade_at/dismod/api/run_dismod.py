import subprocess
import sys
from types import SimpleNamespace
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def run_dismod(dm_file, command):
    """
    Runs a command on a dismod file.
    :param dm_file: (str) the dismod db filepath
    :param command: (str) a command to run
    :return:
    """
    command = ["dmdismod", str(dm_file), command]
    command = ' '.join(command)
    LOG.info(f"Running {command}...")

    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    info = SimpleNamespace()
    info.exit_status = process.returncode
    info.stdout = process.stdout.decode()
    info.stderr = process.stderr.decode()

    return info


def run_dismod_commands(dm_file, commands):
    """
    Runs multiple commands on a dismod file and returns the exit statuses.
    Will raise an exception if it runs into an error.

    Args:
        dm_file: (str) the dismod db filepath
        commands: (List[str]) a list of strings

    """
    if isinstance(commands, str):
        commands = [commands]
    for c in commands:
        process = run_dismod(dm_file=dm_file, command=c)
        if process.exit_status:
            LOG.error(f"{c} failed with exit_status {process.exit_status}:")
            LOG.error(f"Error: {process.stderr}")
            LOG.error(f"Output: {process.stdout}")
            try:
                raise RuntimeError(
                    f"Dismod-AT failed with exit status {process.exit_status}."
                    f"Exiting program."
                )
            except RuntimeError:
                sys.exit(process.exit_status)
        else:
            LOG.info(f"{process.stdout}")
            LOG.info(f"{process.stderr}")

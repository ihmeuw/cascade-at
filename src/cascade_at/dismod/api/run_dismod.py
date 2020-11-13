import subprocess
import sys
from types import SimpleNamespace
from typing import List
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def run_dismod(dm_file: str, command: str):
    """
    Executes a command on a dismod file.

    Parameters
    ----------
    dm_file
        the dismod db filepath
    command
        a command to run
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


def run_dismod_commands(dm_file: str, commands: List[str], sys_exit=True):
    """
    Runs multiple commands on a dismod file and returns the exit statuses.
    Will raise an exception if it runs into an error.

    Parameters
    ----------
    dm_file
        the dismod db filepath
    commands
        a list of strings
    sys_exit
        whether to exit the code altogether if there is an error. If False,
        then it will pass the error string back to the original python process.
    """
    processes = dict()
    if isinstance(commands, str):
        commands = [commands]
    for c in commands:
        process = run_dismod(dm_file=dm_file, command=c)
        processes.update({c: process})
        if process.exit_status:
            LOG.error(f"{c} failed with exit_status {process.exit_status}:")
            LOG.error(f"Error: {process.stderr}")
            LOG.error(f"Output: {process.stdout}")
            if sys_exit:
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
    return processes

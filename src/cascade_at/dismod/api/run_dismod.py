import subprocess
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
    process = subprocess.run(command, capture_output=True)
    return {
        "exit_status": process.returncode,
        "stdout": process.stdout.decode(),
        "stderr": process.stderr.decode()
    }

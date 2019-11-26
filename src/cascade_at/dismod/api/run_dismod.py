import subprocess
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
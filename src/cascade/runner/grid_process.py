from functools import lru_cache
from subprocess import run, PIPE, TimeoutExpired, CalledProcessError

from cascade.core.log import getLoggers
from cascade.executor.execution_context import application_config

CODELOG, MATHLOG = getLoggers(__name__)

BLOCK_QCALLS = False
"""
This module-level global guards against writing unit tests that run
on the cluster but fail to run under Travis off the cluster.
Look in tests/conftest.py for implementation.
"""


@lru_cache(maxsize=16)
def find_full_path(executable):
    """Uses the Bash shell's ``which`` command to find the full path
    to the given command. We could hard-code the command location.
    This ``which`` is necessary because grid engine's commands change
    how they pipe to stdout when they sense they are inside a shell.
    """
    process_result = run(
        f"which {executable}", shell=True, stdout=PIPE,
        universal_newlines=True)
    if process_result.returncode != 0:
        # This propagates to application failure. Good reason to restart.
        raise RuntimeError(f"Cannot find {executable} on system")
    return process_result.stdout.strip()


def run_check(executable, arguments):
    """
    Run a process using a set of rules around when to throw an
    exception. We define this here so that there is consistency around
    calling qsub and friends.

    Args:
        executable (str): Either ``qsub``, ``qstat``, ``qconf``, ``qdel``.
        arguments (List): List of arguments. Will be stringified before run.

    Returns:
        str: Standard out as a string.
    """
    if BLOCK_QCALLS:
        raise RuntimeError(
            f"This unit test needs to be marked to run on cluster")
    timeout_key = f"{executable}-timeout-seconds"
    parameters = application_config()["GridEngine"]
    if timeout_key in parameters:
        timeout = int(parameters[timeout_key])
        timeout_failure = parameters["on-failure-timeout-seconds"]
    else:
        CODELOG.info(f"Cannot find key {timeout_key} in GridEngine parameters")
        timeout = 60
        timeout_failure = 600

    # This loop just keeps trying. We will kill it with qdel, if that's
    # what we need.
    while True:
        try:
            executable_path = find_full_path(executable)
            # Requires the full path, or this call will not work.
            process_out = run(
                [str(arg) for arg in [executable_path] + arguments],
                shell=False, universal_newlines=True, stdout=PIPE, stderr=PIPE,
                timeout=timeout, check=True
            )
            return process_out.stdout.strip()
        except CalledProcessError as cpe:
            CODELOG.info(f"{executable} call {cpe.cmd} failed: {cpe.stderr}")
            timeout = timeout_failure
        except TimeoutExpired:
            CODELOG.info(f"{executable} timed out after {timeout}s")
            timeout = timeout_failure

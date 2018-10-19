"""
This module is responsible for running subprocesses according to a plan
that says which must either start or complete before the next can start.
It is a layer over the Python subprocess module.
"""
import os
import subprocess
import time
from types import FunctionType as function

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class NotEnoughResources(Exception):
    """The process runner wasn't given enough resources."""


class ChildProcessProblem(Exception):
    """A subprocess had a nonzero exit code."""


def graph_do(run_next: function, memory_limit: float, sleep_duration: float = 1):
    """
    This runs processes and blocks until completion.
    The ``run_next`` function must have the signature
    ``run_next(completed) -> args.``

    where ``completed`` is a set of the IDs of tasks,
    which are likely location IDs, and it returns a
    dictionary from ID to a Namespace object with attributes
    ``memory`` and ``args``.

    Args:
        run_next: Returns tasks that can run. This function
            has to have semantics of a task graph, meaning it has to return
            results consistent with a fixed set of total tasks and
            one task depends on completion of others, not on any external
            behaviors. The task object this function returns has two
            attributes, ``args``, which is a list of arguments for ``fork``,
            and ``memory`` which is the maximum number of Gb this process
            could require.
        memory_limit: How much memory to use. This function doesn't
            check memory in the operating system. It checks against claims
            by the processes.
        sleep_duration: How long to wait if there's nothing to do.
    """
    completed = set()
    unblocked = run_next(completed)
    running = set()  # Running is a subset of unblocked.
    popen_objects = dict()  # Tracks processes associated with running.

    def _resources_to_run(nr):
        return nr not in running | completed and unblocked[nr].memory <= memory_remaining

    def _process_complete(job_id):
        return popen_objects[job_id].poll() is not None

    # Permit only one process to start or stop on each iteration
    # in order to reduce complexity.
    while unblocked:
        assert running.issubset(set(unblocked.keys()))
        assert not running & completed, f"Run {running} comp {completed}"

        memory_usage = sum(unblocked[run_mem].memory for run_mem in running)
        memory_remaining = memory_limit - memory_usage
        next_to_run = next(filter(_resources_to_run, unblocked.keys()), None)
        if not running and next_to_run is None:
            raise NotEnoughResources(
                f"Processes require more than allotted memory: {memory_limit}. "
                f"Even though {unblocked.keys()} not done and {completed} done."
            )

        if next_to_run is not None:
            # 1) Start a job: U -> U', R -> R' + 1, C -> C'=C
            child = _run_or_throw(unblocked[next_to_run].args)
            running.add(next_to_run)
            popen_objects[next_to_run] = child
        else:
            # 2) Finish a job: U -> U' >= U-1, R -> R'-1, C -> C' + 1
            newly_completed = next(filter(_process_complete, running), None)
            if newly_completed is not None:  # A process ID of 0 is False.
                running.remove(newly_completed)
                completed.add(newly_completed)
                unblocked = run_next(completed)
                CODELOG.debug(f"unblocked new {unblocked.keys()}")
                assert set(unblocked.keys()) not in completed
                result = popen_objects[newly_completed]
                if result.returncode != 0:
                    raise ChildProcessProblem(f"Child {result.args} had return code " f"{result.returncode}")
                del popen_objects[newly_completed]
            else:
                time.sleep(sleep_duration)


def _nice_process():
    os.nice(19)


def _run_or_throw(args):
    try:
        child = subprocess.Popen(args=args, preexec_fn=_nice_process)
    except ValueError as ve:
        raise Exception(f"Invalid arguments to process: {ve}")
    except OSError as ose:
        raise Exception(f"Operating system error running process {ose}")
    return child

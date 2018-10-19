"""
This stage runs Dismod-AT. Dismod gets called in very similar ways.
Let's look at them in order to narrow down configuration of this
stage.::

   dismod_at database init
   dismod_at database fit <variables>
   dismod_at database fit <variables> <simulate_index>
   dismod_at database set option <name> <value>
   dismod_at database set <table_out> <source>
   dismod_at database set <table_out> <source> <sample_index>
   dismod_at database depend
   dismod_at database simulate <number_simulate>
   dismod_at database sample <method> <number_sample>
   dismod_at database predict <source>

So how does the cascade know what the input database is?
We decided it would use the name of the stage as the name of
the database. Can a user call dismod_at through the cascade?
Would they want to? I see no reason for it when you can just
call Dismod. You'd call it within the Cascade when it's a known step,
in which case the variables and sources are decided beforehand.

Therefore, the ``command_list`` below will include the entries
that come after the database.::

   command_list = [
       ["init"],
       ["fit", "both"],
       ["predict"]
   ]

That gives enough freedom to specify the command list when
defining the :class:`cascade_at.sequential_batch.Batch`.
"""
import functools
import os
import asyncio

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class DismodATException(Exception):
    pass


def dismod_run(command_list):
    """
    Returns a batch stage that runs DismodAT on these commands. This
    is a builder. The idea is to use this in ``Batch``. These function
    names are made up, but it shows how to use ``dismod_run``::

       batch = Batch([
           ("settings", import_settings),
           ("init", dismod_run([["init"]])),
           ("country", import_ccov),
           ("fitpredict", dismod_run([
                   ["fit", "fixed"],
                   ["predict"]
                ]))
           ("posteriors", posterior_to_priors)
       ])

    There are two command-line options that affect how DismodAT
    runs.

    * ``single_use_machine=False`` This is True or False where True
      means that we nice the Dismod process in order for it not
      to interfere with interactive work on the same machine.
      This makes the machine much more responsive with little
      loss of efficiency.

    * ``subprocess_poll_time=0.5`` This decides how often to check
      whether DismodAT is done. It is the time in seconds, as
      a floating-point number, to wait between checks. There's
      nothing unreasonable about using a tenth of a second.

    Args:
        command_list (List[List[str]]): A list of commands for DismodAT
            to run. It's the part of the command after the database.

    Returns:
        A callable stage that runs DismodAT on these commands.
    """
    return functools.partial(dismod_recipe, command_list)


def dismod_recipe(command_list, context):
    """
    Runs Dismod-AT. We generally run Dismod-AT more than once with
    a sequence of commands, so we call these a recipe.

    Args:
        command_list (List[List[str]]): A list of commands for Dismod AT.
        context: A context object from which we do I/O.
    """
    dismod_executable = context.dismod_executable()
    # These are checks we can do before trying to run Dismod. They
    # don't need to be exhaustive because we'll see if it doesn't run.
    if len(dismod_executable) < 1:
        raise ValueError("There is no dismod executable in context")
    if not dismod_executable[0].exists():
        raise FileNotFoundError(f"Could not find file {dismod_executable}")
    using_singularity = len(dismod_executable) > 3 and dismod_executable[0].name == "singularity"
    if using_singularity and not dismod_executable[2].exists():
        raise FileNotFoundError(f"Could not find singularity image {dismod_executable[2]}")

    db_file = context.dismod_file()
    if not db_file.exists():
        raise FileNotFoundError(f"Could not find file {db_file}")

    for command in command_list:
        MATHLOG.info("Running dismod_at {} {}".format(db_file, command))
        run_and_watch(
            dismod_executable + [db_file] + command,
            context.params("single_use_machine"),
            context.params("subprocess_poll_time"),
        )


def reduce_process_priority():
    """
    It seems counter-intuitive to ask the process to be slower,
    but reducing the priority of the process makes it livable to run
    in the background on your laptop, and it won't go appreciably
    slower.
    """
    os.nice(19)


@asyncio.coroutine
def _read_pipe(pipe, callback=lambda text: None):
    """Read from a pipe until it closes.

    Args:
        pipe: The pipe to read from
        callback: a callable which will be invoked each time data is read from the pipe
    """
    while not pipe.at_eof():
        text = yield from pipe.read(2 ** 16)
        text = text.decode("utf-8")
        callback(text)


def run_and_watch(command, single_use_machine, poll_time):
    """
    Runs a command and logs its stdout and stderr while that command
    runs. The point is two-fold, to gather stdout from the running
    program and to turn any faults into exceptions.

    Args:
        command (Path|str): The command to run as a rooted path.
        single_use_machine (bool): Whether this is running on a machine
            where someone is doing interactive work at the same time.
            If so, we reduce process priority.
        poll_time (int): How many seconds to wait between checking
            whether the program is done. This isn't an expensive
            operation.

    Returns:
        str: The output stream.
        str: The error stream.
    """
    command = [str(a) for a in command]
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_async_run_and_watch(command, single_use_machine, poll_time))


@asyncio.coroutine
def _async_run_and_watch(command, single_use_machine, poll_time):
    if single_use_machine:
        pre_execution_function = reduce_process_priority
    else:
        pre_execution_function = None

    try:
        CODELOG.info(f"Forking to {command}")
        sub_process = yield from asyncio.subprocess.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, preexec_fn=pre_execution_function
        )
    except ValueError as ve:
        raise Exception(f"Dismod called with invalid arguments {ve}")
    except OSError as ose:
        raise Exception(f"Dismod couldn't run due to OS error {ose}")

    loop = asyncio.get_event_loop()
    std_out_task = loop.create_task(_read_pipe(sub_process.stdout, lambda text: MATHLOG.info(text)))
    std_err_task = loop.create_task(_read_pipe(sub_process.stderr, lambda text: MATHLOG.error(text)))
    yield from sub_process.wait()
    yield from std_out_task
    yield from std_err_task

    if sub_process.returncode != 0:
        msg = f"return code {sub_process.returncode}\n"
        raise DismodATException("dismod_at failed.\n{}".format(msg))
    else:
        pass  # Return code is 0. Success.

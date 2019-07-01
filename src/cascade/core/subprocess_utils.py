import asyncio
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from subprocess import run, DEVNULL
from tempfile import gettempdir
from uuid import uuid4

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)

_NONWHITESPACE = re.compile(r"\S")
MODEL_NAME = re.compile(r"model name\s+: (.*)")


async def _read_pipe(pipe, callback=lambda text: None):
    """Read from a pipe until it closes.

    Args:
        pipe: The pipe to read from
        callback: a callable which will be invoked each time data is read from the pipe
    """
    while not pipe.at_eof():
        text = await pipe.read(2 ** 16)
        text = text.decode("utf-8")
        callback(text)


def _dismod_report_info(accumulator):
    """This ensures MATHLOG messages have a function name in the log.
    Otherwise, they show <lambda> as the function name.
    """

    def inner(text):
        MATHLOG.info(text, extra=dict(is_dismod_output=True))
        accumulator.append(text)

    return inner


def _dismod_report_stderr(accumulator):
    """This ensures MATHLOG messages have a function name in the log.
    Otherwise, they show <lambda> as the function name.
    """

    def inner(text):
        if re.search(_NONWHITESPACE, text):
            MATHLOG.warning(text, extra=dict(is_dismod_output=True))
            accumulator.append(text)

    return inner


async def async_run_with_logging(command, loop):
    sub_process = await asyncio.subprocess.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout_accumulator = []
    stderr_accumulator = []
    std_out_task = loop.create_task(_read_pipe(sub_process.stdout, _dismod_report_info(stdout_accumulator)))
    std_err_task = loop.create_task(_read_pipe(sub_process.stderr, _dismod_report_stderr(stderr_accumulator)))

    await sub_process.wait()
    await std_out_task
    await std_err_task

    return sub_process.returncode, "".join(stdout_accumulator), "".join(stderr_accumulator)


def run_with_logging(command):
    """
    This runs a Unix subprocess. It captures standard out from that subprocess
    while it runs and can print that standard out to a file. It does this
    complicated maneuver because users of Dismod-AT can see its progress
    in a web browser by refreshing a web page that displays the file.

    Args:
        command (List[str]): The command to run, as a list of strings.

    Returns:
        (int, str, str): The return code, stdout, and stderr.
    """
    # This intentionally uses the asyncio supplied by Python 3.6, instead
    # of the newer keywords in Python 3.7, because a prototype implementation
    # with the newer keywords didn't work better and put pressure on using
    # newer versions for testing.
    loop = asyncio.get_event_loop()
    if loop.is_running():
        process = subprocess.run(command, capture_output=True)
        return process.returncode, process.stdout.decode(), process.stderr.decode()
    else:
        return loop.run_until_complete(async_run_with_logging(command, loop))


def processor_type():
    """Gets processor type so that we can adjust for inherent processor
    speed, if that becomes important.

    Returns:
        str: The processor name in a long form, with GHz usually.
    """
    model_name = None
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        with cpu_info.open() as cpu_stream:
            cpu_lines = cpu_stream.read()
        model_name_match = MODEL_NAME.search(cpu_lines)
        if model_name_match:
            model_name = model_name_match.group(1)
    return model_name


@lru_cache(maxsize=1)
def this_machine_has_newer_time_command():
    """Check whether the timer could work. Do this once.
    Some systems have older copies of /usr/bin/time that don't have -v."""
    timer = Path("/usr/bin/time")
    time_line = [str(timer), "-vo"]
    if timer.exists():
        result = run(time_line + ["/dev/null", "/bin/ls"],
                     stdout=DEVNULL, stderr=DEVNULL)
        if result.returncode == 0:
            return time_line
    return None


def add_gross_timing(command):
    """Uses /usr/bin/time to add timing to a command.
    Requires /usr/bin/time on the machine, with a version that has
    the verbose flag and output-to-file flag. It tests whether that's
    present and does no timing if it isn't.
    """
    time_line = this_machine_has_newer_time_command()
    if time_line:
        if isinstance(command, str):
            command = command.split()
        tmp = Path(gettempdir()) / str(uuid4())
        return time_line + [str(tmp)] + command, tmp
    else:
        return command, None


def read_gross_timing(tmp_file):
    """Reads the temporary file with timing information and then deletes it.

    Returns:
        Dictionary of key-value pairs with timing information.
    """
    try:
        with tmp_file.open() as tmp_stream:
            lines = tmp_stream.readlines()
    except OSError as ose:
        CODELOG.info(f"Could not read timing file {ose}")
        return dict()
    try:
        tmp_file.unlink()
    except OSError as ose:
        CODELOG.info(f"Could not delete timing file {ose}")
        return dict()
    key_value = ([x.strip() for x in line.split(": ")] for line in lines)
    return dict(kv for kv in key_value if len(kv) == 2)

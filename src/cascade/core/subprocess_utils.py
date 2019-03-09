import asyncio
import subprocess
import re

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)

_NONWHITESPACE = re.compile(r"\S")


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
    loop = asyncio.get_event_loop()
    if loop.is_running():
        process = subprocess.run(command, capture_output=True)
        return process.returncode, process.stdout.decode(), process.stderr.decode()
    else:
        return loop.run_until_complete(async_run_with_logging(command, loop))

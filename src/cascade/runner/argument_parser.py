"""
Parse command-line arguments.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

from cascade.core.log import getLoggers
from cascade.runner.application_config import application_config

CODELOG, MATHLOG = getLoggers(__name__)


class ArgumentException(Exception):
    """The command-line arguments were wrong."""


class DMArgumentParser(ArgumentParser):
    """
    This argument parser has default arguments in order to make
    it simpler to build multiple main programs. They include

    * ``-v`` for verbose.

    * ``-q`` for quiet. One gets you to warning level, two ``-q`` gets you
      error level, and three ``-q`` turns off exception logging.

    * ``--logmod module_name`` This turns on debug logging for one module.
      For example, ``--logmod cascade.db`` turns on database logging.
      This can be used for several modules.

    * ``--modlevel level_str`` One of debug, info, warning, error, exception.
      This is the logging level for the specific logmod modules and defaults
      to debug.

    * ``stage`` This positional argument lets you specify what stage
      to run among the stages. If it isn't present, then all stages run.

    The two log files are in the Epi directories under

         * ``dismod_at/<mvid>/log.log`` for
           the math log that EpiViz sees.
         * ``at_cascade/logs/<date-time>.log`` for the code log.

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        config = application_config()["DataLayout"]
        root_dir = Path(config["root-directory"]).resolve()
        code_log = config["code-log-directory"]
        epiviz_log = config["epiviz-log-directory"]
        log_parse = self.add_argument_group(
            "logs",
            "Options that affect logging",
        )
        log_parse.add_argument("-v", "--verbose", action="count", default=0,
                               help="Increase verbosity of logging")
        log_parse.add_argument("-q", "--quiet", action="count", default=0,
                               help="Decrease verbosity of logging")
        log_parse.add_argument("--logmod", action="append", default=[],
                               help="Set logging to debug for submodule")
        log_parse.add_argument("--modlevel", type=str, default="debug",
                               help="Log level for specified modules")
        log_parse.add_argument("--epiviz-log", type=Path, default=epiviz_log,
                               help="Directory for EpiViz log")
        log_parse.add_argument("--code-log", type=Path, default=code_log,
                               help="Directory for code log")
        log_parse.add_argument("--root-dir", type=Path, default=root_dir,
                               help="Directory to use as root for logs.")

    def exit(self, status=0, message=None):
        """
        Overrides parent exit because parent does sys.exit.
        This version instead raises an exception which we can catch,
        use to write an error, and then exit.
        """
        if status == 0:
            sys.exit(status)

        elif message:
            super()._print_message(message, sys.stderr)

        else:
            message = "Exiting due to bad arguments: {}".format(sys.argv)

        CODELOG.error(message)
        raise ArgumentException(message, status)

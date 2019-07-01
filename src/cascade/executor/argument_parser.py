"""
Parse command-line arguments.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

import pkg_resources
import toml

from cascade.core.log import getLoggers
from cascade.executor.execution_context import application_config

CODELOG, MATHLOG = getLoggers(__name__)


class ArgumentException(Exception):
    """The command-line arguments were wrong."""


class BaseArgumentParser(ArgumentParser):
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
        self.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity of logging")
        self.add_argument("-q", "--quiet", action="count", default=0, help="Decrease verbosity of logging")
        self.add_argument("--logmod", action="append", default=[], help="Set logging to debug for submodule")
        self.add_argument("--modlevel", type=str, default="debug", help="Log level for specified modules")
        self.add_argument("--epiviz-log", type=Path, default=epiviz_log, help="Directory for EpiViz log")
        self.add_argument("--code-log", type=Path, default=code_log, help="Directory for code log")
        self.add_argument("--root-dir", type=Path, default=root_dir,
                          help="Directory to use as root for logs.")

        arguments = toml.loads(pkg_resources.resource_string("cascade.executor", "data/parameters.toml").decode())
        arg_types = dict(bool=bool, str=str, float=float, int=int)
        for arg_name, spec in arguments.items():
            if spec["type"] == "bool":
                action = "store_{}".format(str(not spec["default"]).lower())
                self.add_argument(f"--{arg_name}", action=action, help=spec["help"])
            else:
                base = dict(default=None, help=None)
                base.update(spec)
                self.add_argument(
                    f"--{arg_name}", type=arg_types[base["type"]], default=base["default"], help=base["help"]
                )

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


class DMArgumentParser(BaseArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("stage", type=str, nargs="?", help="A single stage to run.")

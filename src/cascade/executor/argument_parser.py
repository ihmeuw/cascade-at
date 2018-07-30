"""
Parse command-line arguments.
"""
from argparse import ArgumentParser
import datetime
from getpass import getuser
import logging
import logging.handlers
from pathlib import Path
import sys
import toml

import pkg_resources


CODELOG = logging.getLogger(__name__)
EPIVIZ_LOG_DIR = Path("/ihme/epi/dismod_at/logs")
CODE_LOG_DIR = Path("/ihme/temp/sgeoutput")


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
      For example, ``--logmod cascade_at.db`` turns on database logging.
      This can be used for several modules.

    * ``--modlevel level_str`` One of debug, info, warning, error, exception.
      This is the logging level for the specific logmod modules and defaults
      to debug.

    * ``stage`` This positional argument lets you specify what stage
      to run among the stages. If it isn't present, then all stages run.

    The two log files are

         * ``/ihme/epi/dismod_at/<mvid>/log.log`` for
           the math log that EpiViz sees.
         * ``/ihme/epi/at_cascade/logs/<date-time>.log`` for the code log.

    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_argument("-v", "--verbose", action="count", default=0,
                          help="Increase verbosity of logging")
        self.add_argument("-q", "--quiet", action="count", default=0,
                          help="Decrease verbosity of logging")
        self.add_argument("--logmod", action="append", default=[],
                          help="Set logging to debug for submodule")
        self.add_argument("--modlevel", type=str, default="debug",
                          help="Log level for specified modules")

        arguments = toml.loads(pkg_resources.resource_string(
            "cascade.executor", "data/parameters.toml").decode())
        arg_types = dict(
            bool=bool, str=str, float=float, int=int
        )
        for arg_name, spec in arguments.items():
            if spec["type"] == "bool":
                action = "store_{}".format(str(not spec["default"]).lower())
                self.add_argument(
                    f"--{arg_name}", action=action,
                    help=spec["help"]
                )
            else:
                base = dict(default=None, help=None)
                base.update(spec)
                self.add_argument(
                    f"--{arg_name}", type=arg_types[base["type"]],
                    default=base["default"], help=base["help"]
                )

    def parse_known_args(self, args=None, namespace=None):
        """Parses command line arguments. Use this instead of
        `parse_args` because it's less likely to fail unexpectedly.

        Before doing what ArgumentParser.parse_args normally does, we gotta do
        the provenance.
        """
        _args, _argv = super().parse_known_args(args, namespace)

        self._logging_config(_args)
        return _args, _argv

    def exit(self, status=0, message=None):
        """
        Overrides parent exit because parent does sys.exit.
        This version instead raises an exception which we can catch,
        use to write an error, and then exit.
        """
        if message:
            super()._print_message(message, sys.stderr)
        else:
            message = "Exiting due to bad arguments: {}".format(sys.argv)
        CODELOG.error(message)
        raise ArgumentException(message, status)

    @staticmethod
    def _logging_config(
            args, epiviz_log_dir=EPIVIZ_LOG_DIR, code_log_dir=CODE_LOG_DIR
    ):
        """Configures logging. Command-line arguments ``-v`` and ``-q``
        set both the code log file and what streams to stdout. The ``-v``
        flag turns on debug level, and ``-q`` sets it to info level.
        The math logger will always be set to info level.

        Arguments:
            args (argparse.Namespace): the arguments parsed by self.
            epiviz_log_dir (Path): Directory into which to put the math log.
            code_log_dir (Path): Directory into which to put the code log.
        """
        # Any handlers from from a basicConfig, which we will reconfigure.
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

        level = logging.INFO - 10 * args.verbose + 10 * args.quiet
        fmt = ("%(levelname)s %(asctime)s %(name)s %(funcName)s: "
               "%(message)s")
        datefmt = "%y-%m-%d %H:%M:%S"

        # The command-line logging level specifies what goes to stderr.
        root_handler = logging.StreamHandler(sys.stderr)
        root_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logging.root.addHandler(root_handler)
        logging.root.setLevel(level)

        # If the script is started with a model version ID, then it's
        # probably being run under EpiViz and should make a math log.
        if args.mvid is not None and epiviz_log_dir.exists():
            mvid = args.mvid.lower()
            math_log_dir = epiviz_log_dir / mvid
            try:
                math_log_dir.mkdir()
            except FileExistsError:
                pass
            math_handler = logging.FileHandler(str(math_log_dir / "log.log"))
            math_handler.setFormatter(logging.Formatter(fmt, datefmt))
            math_logger = logging.getLogger("cascade_at.math")
            math_logger.addHandler(math_handler)
            math_logger.setLevel(logging.INFO)

        if code_log_dir.exists():
            user_code_dir = code_log_dir / getuser() / "cascade"
            user_code_dir.mkdir(parents=True, exist_ok=True)
            fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.log")
            code_log_path = user_code_dir / fname
            code_handler = logging.FileHandler(str(code_log_path))
            code_handler.setFormatter(logging.Formatter(fmt, datefmt))
            # The memory handler reduces the number of writes to disk.
            # It will flush writes when it encounters an ERROR.
            outer_handler = logging.handlers.MemoryHandler(
                capacity=128000, target=code_handler)
            logging.root.addHandler(outer_handler)
            logging.getLogger("cascade_at").setLevel(level)

        if args.logmod:
            module_log_level = getattr(logging, args.modlevel.upper(),
                                       args.modlevel)
            if not isinstance(module_log_level, int):
                try:
                    module_log_level = int(module_log_level)
                except ValueError:
                    logging.warning("Could not parse modlevel {}".format(
                        args.modlevel))
                    module_log_level = logging.DEBUG

            for submodule in args.logmod:
                logging.getLogger(submodule).setLevel(module_log_level)


class DMArgumentParser(BaseArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("stage", type=str, nargs="?",
                          help="A single stage to run.")

"""
Parse command-line arguments.
"""
from argparse import ArgumentParser
import datetime
from getpass import getuser
import logging.handlers
import os
from pathlib import Path
import sys
import toml
import tempfile

import pkg_resources

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)

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
      For example, ``--logmod cascade.db`` turns on database logging.
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

        self.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity of logging")
        self.add_argument("-q", "--quiet", action="count", default=0, help="Decrease verbosity of logging")
        self.add_argument("--logmod", action="append", default=[], help="Set logging to debug for submodule")
        self.add_argument("--modlevel", type=str, default="debug", help="Log level for specified modules")
        self.add_argument("--epiviz-log", type=Path, default=EPIVIZ_LOG_DIR, help="Directory for EpiViz log")
        self.add_argument("--code-log", type=Path, default=CODE_LOG_DIR, help="Directory for code log")

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

    def parse_args(self, args=None, namespace=None):
        """Parses command line arguments. Use this instead of
        `parse_known_args` because it won't fail silently.

        Before doing what ArgumentParser.parse_args normally does, we gotta do
        the provenance.
        """
        _args = super().parse_args(args, namespace)

        self._logging_config(_args)
        return _args

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
    def _logging_config(args):
        """Configures logging. Command-line arguments ``-v`` and ``-q``
        set both the code log file and what streams to stdout. The ``-v``
        flag turns on debug level, and ``-q`` sets it to info level.
        The math logger will always be set to debug level.

        Arguments:
            args (argparse.Namespace): the arguments parsed by self. This
                must have members ``verbose``, ``quiet``, ``code_log``,
                ``mvid``, ``epiviz_log``, ``logmod``, and ``modlevel``.
        """
        # Any handlers from a basicConfig, which we will reconfigure.
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

        level = logging.INFO - 10 * args.verbose + 10 * args.quiet
        # The command-line logging level specifies what goes to stderr.
        root_handler = logging.StreamHandler(sys.stderr)
        fmt = "%(levelname)s %(asctime)s %(pathname)s:%(lineno)d: %(message)s"
        datefmt = "%y-%m-%d %H:%M:%S"
        root_handler.setFormatter(logging.Formatter(fmt, datefmt))
        root_handler.setLevel(level)
        logging.root.addHandler(root_handler)
        logging.root.setLevel(level)

        code_log = BaseArgumentParser._logging_configure_root_log(args.code_log, level)
        BaseArgumentParser._logging_configure_mathlog(args.mvid, args.epiviz_log)
        BaseArgumentParser._logging_individual_modules(args.logmod, args.modlevel)
        if code_log:  # Tell the math log people where the code log is located.
            logging.getLogger("cascade.math").info(f"Code log is at {code_log}")

    @staticmethod
    def _logging_configure_root_log(code_log_dir, level):
        user_code_dir = code_log_dir / getuser() / "cascade"
        try:
            user_code_dir_exists = user_code_dir.exists()
        except (OSError, PermissionError) as uce:
            logging.warning(f"Cannot read user code dir {user_code_dir}: {uce}")
            return
        if not user_code_dir_exists:
            try:
                user_code_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError as pe:
                logging.warning(f"Could not write to {user_code_dir} {pe}. "
                                f"Not making a log file for code log.")
                return

        fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            log_file_descriptor, code_log_path = tempfile.mkstemp(
                suffix=".log", prefix=fname, dir=user_code_dir, text=True)
        except (OSError, PermissionError) as ose:
            logging.warning(f"Could not write to file in {user_code_dir} to make a log. {ose}")
            return

        os.close(log_file_descriptor)
        try:
            code_handler = logging.FileHandler(code_log_path)
        except (OSError, PermissionError) as fhe:
            logging.warning(f"Could not write to file {code_log_path} so no code log: {fhe}")
            return
        code_handler.setLevel(level)
        fmt = "%(levelname)s %(asctime)s %(pathname)s:%(lineno)d: %(message)s"
        datefmt = "%y-%m-%d %H:%M:%S"
        code_handler.setFormatter(logging.Formatter(fmt, datefmt))
        # The memory handler reduces the number of writes to disk.
        # It will flush writes when it encounters an ERROR.
        outer_handler = logging.handlers.MemoryHandler(capacity=128000,
                                                       target=code_handler)
        outer_handler.setLevel(level)
        logging.root.addHandler(outer_handler)
        return code_log_path

    @staticmethod
    def _logging_configure_mathlog(mvid, epiviz_log_dir):
        # If the script is started with a model version ID, then it's
        # probably being run under EpiViz and should make a math log.
        if mvid is None:
            logging.warning(f"There is no mvid, so will not write a mathlog.")
            return
        try:
            if not epiviz_log_dir.exists():
                logging.warning(f"There is no epiviz log dir {epiviz_log_dir} so not writing math log.")
                return
        except (OSError, PermissionError) as ele:
            logging.warning(f"Could not read epiviz log dir due to permissions {epiviz_log_dir} {ele}")
            return

        math_log_dir = epiviz_log_dir / str(mvid)
        try:
            if not math_log_dir.exists():
                math_log_dir.mkdir()
        except (OSError, PermissionError) as ose:
            logging.warning(f"Could not make mathlog directory {math_log_dir} "
                            f"even though epiviz log dir {epiviz_log_dir} exists: {ose}")
            return

        log_file = math_log_dir / "log.log"
        try:
            append_to_math_log = "a"
            math_handler = logging.FileHandler(str(log_file), append_to_math_log)
        except (OSError, PermissionError) as mhe:
            logging.warning(f"Could not write to math log at {log_file} even though "
                            f"directory {math_log_dir} exists: {mhe}")
            return
        fmt = "%(levelname)s %(asctime)s %(funcName)s: %(message)s"
        datefmt = "%y-%m-%d %H:%M:%S"
        math_handler.setFormatter(logging.Formatter(fmt, datefmt))
        math_handler.setLevel(logging.DEBUG)
        math_logger = logging.getLogger(f"{__name__.split('.')[0]}.math")
        math_logger.addHandler(math_handler)
        math_logger.setLevel(logging.DEBUG)
        logging.getLogger("cascade.math").info(f"EpiViz log is at {log_file}")

    @staticmethod
    def _logging_individual_modules(logmod, modlevel):
        """Set a list of modules to a particular logging level."""
        if not logmod: return

        module_log_level = getattr(logging, modlevel.upper(), modlevel)
        if not isinstance(module_log_level, int):
            try:
                module_log_level = int(module_log_level)
            except ValueError:
                logging.warning("Could not parse modlevel {}".format(modlevel))
                module_log_level = logging.DEBUG

        for submodule in logmod:
            logging.getLogger(submodule).setLevel(module_log_level)


class DMArgumentParser(BaseArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("stage", type=str, nargs="?", help="A single stage to run.")

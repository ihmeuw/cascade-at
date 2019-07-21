import datetime
import logging
import logging.handlers
import os
import sys
from getpass import getuser
from secrets import token_urlsafe

from cascade.runner.math_log import MathLogFormatter


def logging_config(args):
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

    code_log = _logging_configure_root_log(args.root_dir / args.code_log, level)
    mvid = args.mvid if hasattr(args, "mvid") else "mvid"
    _logging_configure_mathlog(mvid, args.root_dir / args.epiviz_log)
    _logging_individual_modules(args.logmod, args.modlevel)
    if code_log:  # Tell the math log people where the code log is located.
        logging.getLogger("cascade.math").info(f"Code log is at {code_log}")


def _logging_configure_root_log(code_log_dir, level):
    user_code_dir = code_log_dir / getuser() / "dismod"
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

    if not os.access(str(user_code_dir), os.W_OK):
        logging.warning(f"Could not write to file in {user_code_dir} to make a log")
        return

    # On collision, this will fail to write a code log.
    # Not using tempfile because it insists on secure mode flags.
    fname = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S{token_urlsafe(8)}.log")
    code_log_path = user_code_dir / fname
    try:
        code_handler = logging.StreamHandler(
            open(os.open(str(code_log_path), os.O_CREAT | os.O_WRONLY, 0o644), "w"))
    except (OSError, PermissionError) as fhe:
        logging.warning(f"Could not write to file {code_log_path} so no code log: {fhe}")
        return
    code_handler.setLevel(level)
    fmt = "%(levelname)s %(asctime)s %(pathname)s:%(lineno)d: %(message)s"
    datefmt = "%y-%m-%d %H:%M:%S"
    code_handler.setFormatter(logging.Formatter(fmt, datefmt))
    logging.root.addHandler(code_handler)
    return code_log_path


def _logging_configure_mathlog(mvid, epiviz_log_dir):
    """
    The mathlog is shown in the EpiViz GUI in a web browser, so it has
    less detail about the code. On error, we can check the CODELOG.

    Args:
        mvid: If the script is started with a model version ID, then it's
            probably being run under EpiViz and should make a math log.
        epiviz_log_dir (Path): Directory in which to make the file.

    """
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
    if not os.access(str(math_log_dir), os.W_OK):
        logging.warning(f"Could not write log to {math_log_dir}")
        return

    log_file = math_log_dir / "log.log"
    try:
        math_handler = logging.StreamHandler(open(os.open(str(log_file), os.O_CREAT | os.O_WRONLY, 0o644), "w"))
    except (OSError, PermissionError) as mhe:
        logging.warning(f"Could not write to math log at {log_file} even though "
                        f"directory {math_log_dir} exists: {mhe}")
        return
    # The br is an HTML tag to add a line break.
    math_handler.setFormatter(MathLogFormatter())
    math_handler.setLevel(logging.DEBUG)
    math_logger = logging.getLogger(f"{__name__.split('.')[0]}.math")
    math_logger.addHandler(math_handler)
    math_logger.setLevel(logging.DEBUG)
    logging.getLogger("cascade.math").info(f"EpiViz log is at {log_file}")


def _logging_individual_modules(logmod, modlevel):
    """Set a list of modules to a particular logging level."""
    if not logmod:
        return

    module_log_level = getattr(logging, modlevel.upper(), modlevel)
    if not isinstance(module_log_level, int):
        try:
            module_log_level = int(module_log_level)
        except ValueError:
            logging.warning("Could not parse modlevel {}".format(modlevel))
            module_log_level = logging.DEBUG

    for submodule in logmod:
        logging.getLogger(submodule).setLevel(module_log_level)

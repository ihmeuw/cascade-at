import getpass
import logging
import os
import stat
from pathlib import Path

import pytest
from gridengineapp import ArgumentError

from cascade.executor.dismodel_main import DismodAT
from cascade.runner.cascade_logging import logging_config


def test_argparse_happy():
    parser = DismodAT.add_arguments()
    args = parser.parse_args(["-v"])
    assert args.verbose == 1


def test_argparse_functions():
    parser = DismodAT.add_arguments()
    args = parser.parse_args([])
    assert args.quiet == 0
    assert not args.logmod


def test_argparse_quiet():
    parser = DismodAT.add_arguments()
    args = parser.parse_args(["-q"])
    assert args.verbose == 0
    assert args.quiet == 1


def test_argparse_fail():
    parser = DismodAT.add_arguments()
    with pytest.raises(ArgumentError, match="expected one argument"):
        parser.parse_args(["--hiya", "there", "--logmod"])


def close_all_handlers():
    """Close handlers in order to ensure they have written files."""
    loggers = [logging.root, logging.getLogger("cascade"), logging.getLogger("cascade.math")]
    for logger in loggers:
        for close_root in logger.handlers:
            close_root.flush()
            close_root.close()


def test_math_log(tmpdir):
    tmp_dir = Path(tmpdir)
    parser = DismodAT.add_arguments()
    mvid = "32768"
    # The math log will ignore the -v here, which sets others to DEBUG level.
    args = parser.parse_args([
        "-v", "--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir),
        "--mvid", mvid,  # Must have an mvid for math log to happen.
    ])
    logging_config(args)
    mathlog = logging.getLogger("cascade.math.complicated")
    mathlog.debug("debugfi")
    mathlog.info("infofum")
    mathlog.warning("warningfo")
    mathlog.error("errorhum")
    close_all_handlers()

    in_log = (tmp_dir / mvid / "log.log").read_text().splitlines()
    print(f"math log is ========={in_log}==============")
    assert any("infofum" in in_line for in_line in in_log)
    assert any("warningfo" in in_line for in_line in in_log)
    assert any("errorhum" in in_line for in_line in in_log)
    # Math log is set to DEBUG level.
    assert any("debugfi" in in_line for in_line in in_log)


def test_code_log(tmpdir):
    previous_umask = os.umask(0o002)
    tmp_dir = Path(tmpdir)
    parser = DismodAT.add_arguments()
    args = parser.parse_args([
        "--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir),
    ])
    logging_config(args)

    codelog = logging.getLogger("cascade.whatever.complicated")
    codelog.debug("debugfil")
    codelog.info("infofuml")
    codelog.warning("warningfol")
    codelog.error("errorhuml")
    close_all_handlers()

    os.umask(previous_umask)

    base_dir = tmp_dir / getpass.getuser() / "dismod"
    logs = list(base_dir.glob("*.log"))
    print(logs)
    code_log = logs[0].read_text().splitlines()
    # It must be world-readable
    assert os.stat(logs[0]).st_mode & stat.S_IROTH > 0
    assert any("infofuml" in in_line.strip() for in_line in code_log)
    assert any(in_line.strip().endswith("warningfol") for in_line in code_log)
    assert any(in_line.strip().endswith("errorhuml") for in_line in code_log)
    assert not any(in_line.strip().endswith("debugfil") for in_line in code_log)


def test_reduced_code_log(tmpdir):
    tmp_dir = Path(tmpdir)
    parser = DismodAT.add_arguments()
    args = parser.parse_args(
        ["-q", "--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir)])
    logging_config(args)

    codelog = logging.getLogger("cascade.whatever.complicated")
    codelog.debug("debugfic")
    codelog.info("infofumc")
    codelog.warning("warningfoc")
    codelog.error("errorhumc")
    close_all_handlers()

    base_dir = tmp_dir / getpass.getuser() / "dismod"
    code_log = next(base_dir.glob("*.log")).read_text().splitlines()
    assert any(in_line.strip().endswith("warningfoc") for in_line in code_log)
    assert any(in_line.strip().endswith("errorhumc") for in_line in code_log)
    assert not any(in_line.strip().endswith("infofumc") for in_line in code_log)
    assert not any(in_line.strip().endswith("debugfic") for in_line in code_log)


def test_math_log_fail_bad_dir(tmpdir, capsys):
    tmp_dir = Path(tmpdir)
    parser = DismodAT.add_arguments()
    mvid = "191009"
    args = parser.parse_args([
        "--epiviz-log", "bogus", "--code-log", str(tmp_dir),
        "--mvid", mvid,  # must have an mvid
    ])
    logging_config(args)
    close_all_handlers()

    logging.getLogger("cascade.whatever.complicated")
    assert "no epiviz log dir" in capsys.readouterr().err


def test_math_log_fail_subdir_fail(tmpdir, capsys):
    tmp_dir = Path(tmpdir)
    mvid = "191009"
    ez_log = tmp_dir / "ezlog"
    ez_log.mkdir(mode=stat.S_IWUSR)  # This should be a creative failure.
    parser = DismodAT.add_arguments()
    args = parser.parse_args([
        "--epiviz-log", str(ez_log), "--code-log", str(tmp_dir),
        "--mvid", mvid,  # must have an mvid
    ])
    logging_config(args)

    logging.getLogger("cascade.whatever.complicated")
    assert "Could not make" in capsys.readouterr().err

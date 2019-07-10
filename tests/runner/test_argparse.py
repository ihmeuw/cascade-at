import getpass
import logging
import os
from pathlib import Path
import pkg_resources
import pytest
import stat
import toml

from cascade.runner.argument_parser import DMArgumentParser, ArgumentException
from cascade.runner.cascade_logging import logging_config


def test_argparse_happy():
    parser = DMArgumentParser()
    args = parser.parse_args(["-v"])
    assert args.verbose == 1


def test_argparse_functions():
    parser = DMArgumentParser()
    args = parser.parse_args([])
    assert args.quiet == 0
    assert not args.logmod


def test_argparse_quiet():
    parser = DMArgumentParser()
    args = parser.parse_args(["-q"])
    assert args.verbose == 0
    assert args.quiet == 1


def test_argparse_fail():
    parser = DMArgumentParser()
    with pytest.raises(ArgumentException):
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
    parser = DMArgumentParser()
    # The math log will ignore the -v here, which sets others to DEBUG level.
    args = parser.parse_args([
        "-v", "--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir)])
    logging_config(args)
    mathlog = logging.getLogger("cascade.math.complicated")
    mathlog.debug("debugfi")
    mathlog.info("infofum")
    mathlog.warning("warningfo")
    mathlog.error("errorhum")
    close_all_handlers()

    mvid = "mvid"  # because there is no mvid defined here, so it's a default.
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
    parser = DMArgumentParser()
    args = parser.parse_args(["--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir)])
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
    parser = DMArgumentParser()
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
    parser = DMArgumentParser()
    args = parser.parse_args(["--epiviz-log", "bogus", "--code-log", str(tmp_dir)])
    logging_config(args)
    close_all_handlers()

    logging.getLogger("cascade.whatever.complicated")
    assert "no epiviz log dir" in capsys.readouterr().err


def test_math_log_fail_subdir_fail(tmpdir, capsys):
    tmp_dir = Path(tmpdir)
    ez_log = tmp_dir / "ezlog"
    ez_log.mkdir(mode=stat.S_IWUSR)  # This should be a creative failure.
    parser = DMArgumentParser()
    args = parser.parse_args(["--epiviz-log", str(ez_log), "--code-log", str(tmp_dir)])
    logging_config(args)

    logging.getLogger("cascade.whatever.complicated")
    assert "Could not make" in capsys.readouterr().err


def test_parameter_file_proper_toml():
    """Tells you what line of the TOML has an error"""
    ll = pkg_resources.resource_string("cascade.executor", "data/parameters.toml").decode().split("\n")
    for i in range(1, len(ll)):
        try:
            toml.loads("".join(ll[:i]))
        except toml.TomlDecodeError:
            assert False, f"failed on line {i}: {ll[i-1]}"


def test_read_parameters():
    """
    Did you modify the parameters.toml file? This checks that your edits
    didn't sabotage that file.
    """
    arguments = toml.loads(pkg_resources.resource_string("cascade.executor", "data/parameters.toml").decode())
    assert isinstance(arguments, dict)
    assert len(arguments) > 10
    for name, arg in arguments.items():
        assert "type" in arg, f"{name} has no type"
        assert "help" in arg, f"{name} has no help"
        assert "_" not in name  # dashes, not underscores
        if "default" in arg:
            assert type(arg["default"]).__name__ == arg["type"], f"{name} is wrong type"

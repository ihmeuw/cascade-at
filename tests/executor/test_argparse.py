import getpass
import logging
from pathlib import Path
import pkg_resources
import pytest
import stat
import toml

from cascade.executor.argument_parser import DMArgumentParser, ArgumentException


def test_argparse_happy():
    parser = DMArgumentParser()
    args = parser.parse_args(["-v", "importer"])
    assert args.verbose == 1
    assert args.stage == "importer"


def test_argparse_functions():
    parser = DMArgumentParser()
    args = parser.parse_args([])
    assert args.stage is None


def test_argparse_quiet():
    parser = DMArgumentParser()
    args = parser.parse_args(["-q"])
    assert args.verbose == 0
    assert args.quiet == 1
    assert args.stage is None


def test_argparse_fail():
    parser = DMArgumentParser()
    with pytest.raises(ArgumentException):
        args = parser.parse_args(["--hiya", "there", "--logmod"])


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
        "-v", "--mvid", "2347", "--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir)])

    mathlog = logging.getLogger("cascade.math.complicated")
    mathlog.debug("debugfi")
    mathlog.info("infofum")
    mathlog.warning("warningfo")
    mathlog.error("errorhum")
    close_all_handlers()

    in_log = (tmp_dir / str(args.mvid) / "log.log").read_text().splitlines()
    assert any(in_line.strip().endswith("infofum") for in_line in in_log)
    assert any(in_line.strip().endswith("warningfo") for in_line in in_log)
    assert any(in_line.strip().endswith("errorhum") for in_line in in_log)
    # Math log is set to DEBUG level.
    assert any(in_line.strip().endswith("debugfi") for in_line in in_log)


def test_code_log(tmpdir):
    tmp_dir = Path(tmpdir)
    parser = DMArgumentParser()
    args = parser.parse_args(["--mvid", "2347", "--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir)])

    codelog = logging.getLogger("cascade.whatever.complicated")
    codelog.debug("debugfil")
    codelog.info("infofuml")
    codelog.warning("warningfol")
    codelog.error("errorhuml")
    close_all_handlers()

    base_dir = tmp_dir / getpass.getuser() / "cascade"
    logs = list(base_dir.glob("*.log"))
    print(logs)
    code_log = logs[0].read_text().splitlines()
    assert any(in_line.strip().endswith("infofuml") for in_line in code_log)
    assert any(in_line.strip().endswith("warningfol") for in_line in code_log)
    assert any(in_line.strip().endswith("errorhuml") for in_line in code_log)
    assert not any(in_line.strip().endswith("debugfil") for in_line in code_log)


def test_reduced_code_log(tmpdir):
    tmp_dir = Path(tmpdir)
    parser = DMArgumentParser()
    args = parser.parse_args(
        ["-q", "--mvid", "2347", "--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir)])

    codelog = logging.getLogger("cascade.whatever.complicated")
    codelog.debug("debugfic")
    codelog.info("infofumc")
    codelog.warning("warningfoc")
    codelog.error("errorhumc")
    close_all_handlers()

    base_dir = tmp_dir / getpass.getuser() / "cascade"
    code_log = next(base_dir.glob("*.log")).read_text().splitlines()
    assert any(in_line.strip().endswith("warningfoc") for in_line in code_log)
    assert any(in_line.strip().endswith("errorhumc") for in_line in code_log)
    assert not any(in_line.strip().endswith("infofumc") for in_line in code_log)
    assert not any(in_line.strip().endswith("debugfic") for in_line in code_log)


def test_math_log_fail_bad_dir(tmpdir, capsys):
    tmp_dir = Path(tmpdir)
    parser = DMArgumentParser()
    args = parser.parse_args(["--mvid", "2347", "--epiviz-log", "bogus", "--code-log", str(tmp_dir)])
    close_all_handlers()

    codelog = logging.getLogger("cascade.whatever.complicated")
    assert "no epiviz log dir" in capsys.readouterr().err


def test_math_log_fail_no_mvid(tmpdir, capsys):
    tmp_dir = Path(tmpdir)
    parser = DMArgumentParser()
    args = parser.parse_args(["--epiviz-log", str(tmp_dir), "--code-log", str(tmp_dir)])

    codelog = logging.getLogger("cascade.whatever.complicated")
    assert "no mvid" in capsys.readouterr().err


def test_math_log_fail_subdir_fail(tmpdir, capsys):
    tmp_dir = Path(tmpdir)
    ez_log = tmp_dir / "ezlog"
    ez_log.mkdir(mode=stat.S_IWUSR)  # This should be a creative failure.
    parser = DMArgumentParser()
    args = parser.parse_args(["--mvid", "2347", "--epiviz-log", str(ez_log), "--code-log", str(tmp_dir)])

    codelog = logging.getLogger("cascade.whatever.complicated")
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

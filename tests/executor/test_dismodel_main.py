import logging
from getpass import getuser

import cascade.executor.dismodel_main
from cascade.core import getLoggers
from cascade.executor.argument_parser import EPIVIZ_LOG_DIR, CODE_LOG_DIR
from cascade.executor.dismodel_main import entry, main, parse_arguments

CODELOG, MATHLOG = getLoggers(__name__)


def mock_main(args):
    CODELOG.debug("CODELOG debug")
    CODELOG.info("CODELOG info")
    MATHLOG.debug("MATHLOG debug")
    MATHLOG.info("MATHLOG info")


def test_entry_constructs_logs(monkeypatch, tmp_path):
    """Test whether logs are created correctly."""
    # The cases include
    # a) has mvid or not
    # b) directories exist or not
    # c) -v, -q, -v -v
    # Check for g+w for this user on this machine.
    code_dir = tmp_path / CODE_LOG_DIR
    code_dir.mkdir(parents=True)
    math_dir = tmp_path / EPIVIZ_LOG_DIR
    math_dir.mkdir(parents=True)

    monkeypatch.setattr(cascade.executor.dismodel_main, "main", mock_main)
    mvid = "2745"
    args = ["--root-dir", str(tmp_path), "entry_constructs_logs.db", "--mvid", mvid]

    entry(args)

    # Close all of the loggers so that they flush to disk.
    for logger in [logging.root, logging.getLogger("cascade"), logging.getLogger("cascade.math")]:
        for handler in logger.handlers:
            if hasattr(handler, "close"):
                handler.close()

    code_log_dir = code_dir / getuser() / "dismod"
    code_log_list = list(code_log_dir.glob("*.log"))
    assert len(code_log_list) == 1
    code_log = code_log_list[0]

    print(f"code log {code_log}")
    code_lines = code_log.open().readlines()
    assert len(code_lines) > 0

    math_log = math_dir / mvid / "log.log"

    print(f"math log {math_log}")
    math_lines = math_log.open().readlines()
    assert len(math_lines) > 0


def test_main(monkeypatch, ihme):

    def mock_estimate(ec, local_settings):
        pass

    monkeypatch.setattr(cascade.executor.dismodel_main, "estimate_location", mock_estimate)

    args = parse_arguments("z.db --mvid 267737".split())
    main(args)

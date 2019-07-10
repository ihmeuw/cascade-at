import logging
from getpass import getuser

import cascade.runner.entry
from cascade.core import getLoggers
from cascade.runner.application_config import application_config
from cascade.runner.entry import entry

CODELOG, MATHLOG = getLoggers(__name__)


class FakeApp:
    def add_arguments(self, parser):
        parser.add_argument("--mvid", type=int)
        return parser


def mock_main(app, args):
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
    directories = application_config()["DataLayout"]
    code_dir = tmp_path / directories["code-log-directory"]
    code_dir.mkdir(parents=True)
    math_dir = tmp_path / directories["epiviz-log-directory"]
    math_dir.mkdir(parents=True)

    monkeypatch.setattr(cascade.runner.entry, "main", mock_main)
    mvid = "2745"
    args = ["--root-dir", str(tmp_path), "--mvid", mvid]
    app = FakeApp()

    entry(app, args)

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

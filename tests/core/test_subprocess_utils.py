import logging

import pytest

from cascade.core.subprocess_utils import run_with_async_logging


def test_run_with_async_logging__stdout(caplog):
    with caplog.at_level(logging.INFO):
        exit_code, stdout, stderr = run_with_async_logging(["echo", "stdout test"])
    assert exit_code == 0
    assert stdout == "stdout test\n"
    assert stderr == ""
    assert "stdout test" in caplog.text


def test_run_with_async_logging__stderr(caplog):
    exit_code, stdout, stderr = run_with_async_logging(["bash", "-c", "echo stderr test 1>&2"])
    assert exit_code == 0
    assert stdout == ""
    assert stderr == "stderr test\n"
    assert "stderr test" in caplog.text


def test_run_with_async_logging__non_zero_exit():
    exit_code, stdout, stderr = run_with_async_logging(["false"])
    assert exit_code != 0
    assert stdout == ""
    assert stderr == ""


def test_run_with_async_logging__bad_executable():
    with pytest.raises(FileNotFoundError):
        run_with_async_logging(["blargh"])

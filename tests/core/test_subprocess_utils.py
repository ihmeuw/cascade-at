import logging
from subprocess import run

import pytest

from cascade.core.subprocess_utils import (
    run_with_logging, add_gross_timing, read_gross_timing
)


def test_run_with_logging__stdout(caplog):
    with caplog.at_level(logging.INFO):
        exit_code, stdout, stderr = run_with_logging(["echo", "stdout test"])
    assert exit_code == 0
    assert stdout == "stdout test\n"
    assert stderr == ""
    assert "stdout test" in caplog.text


def test_run_with_logging__stderr(caplog):
    exit_code, stdout, stderr = run_with_logging(["bash", "-c", "echo stderr test 1>&2"])
    assert exit_code == 0
    assert stdout == ""
    assert stderr == "stderr test\n"
    assert "stderr test" in caplog.text


def test_run_with_logging__non_zero_exit():
    exit_code, stdout, stderr = run_with_logging(["false"])
    assert exit_code != 0
    assert stdout == ""
    assert stderr == ""


def test_run_with_logging__bad_executable():
    with pytest.raises(FileNotFoundError):
        run_with_logging(["blargh"])


def test_add_gross_timing():
    command = "ls ."
    command, tmp_file = add_gross_timing(command)
    assert command == ["/usr/bin/time", "-vo", str(tmp_file), "ls", "."]


def test_try_gross_timing():
    command, tmp_file = add_gross_timing("/bin/ls .")
    res = run(command)
    assert res.returncode == 0
    kv = read_gross_timing(tmp_file)
    assert len(kv) > 0


def test_read_gross_timing(tmp_path):
    f = tmp_path / "z.txt"
    with f.open("w") as fout:
        fout.write("\tCommand being timed: 'ls'\n")
        fout.write("\tSwaps: 0\n")
        fout.write("Randomness\n")
    kv = read_gross_timing(f)
    assert "Command being timed" in kv
    assert "Swaps" in kv
    assert kv["Swaps"] == "0"
    assert len(kv) == 2

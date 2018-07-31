import logging
import os
from pathlib import Path
import signal
import sys

import pytest

import cascade.executor.dismod_runner as dr


@pytest.fixture
def dmchat():
    return Path(sys.prefix + "/bin/dmchat")


@pytest.fixture
def dmdummy():
    return Path(sys.prefix + "/bin/dmdummy")


def test_run_ls():
    dr.run_and_watch(["/bin/ls"], False, 15)


def test_run_dmchat(dmchat):
    o, e = dr.run_and_watch([dmchat, "5", "0", "0"], False, 2)
    assert o == "".join(["out" + os.linesep] * 5)


def test_dmchat_low_priority(dmchat):
    o, e = dr.run_and_watch([dmchat, "2", "0", "0"], True, 2)
    assert e == "".join(["err" + os.linesep] * 2)


def test_dmchat_nonzero(dmchat):
    with pytest.raises(Exception):
        dr.run_and_watch([dmchat, "1", "0", "7"], True, 2)


@pytest.mark.parametrize("sig_enum", [(signal.SIGINT,), (signal.SIGKILL,), (signal.SIGSEGV,), (signal.SIGSTOP,)])
def test_dmchat_sigint(dmchat, sig_enum):
    with pytest.raises(Exception):
        dr.run_and_watch([dmchat, "2", str(sig_enum.value), "0"], False, 2)


def test_runner_bad_command():
    with pytest.raises(Exception):
        dr.run_and_watch(["bogus"], True, 7)


class RecipeContext:
    def __init__(self, db_file, dm_fake):
        self._db_file = db_file
        self._dismod = [dm_fake]

    def dismod_executable(self):
        return self._dismod

    def dismod_file(self):
        return self._db_file

    def params(self, name):
        return {"single_use_machine": False, "subprocess_poll_time": 2}[name]


def test_recipe_sunny(tmpdir, dmdummy):
    logging.basicConfig(level=logging.DEBUG)
    db_file = tmpdir / "input.db"
    db_file.open("w").close()

    rc = RecipeContext(db_file, dmdummy)
    dr.dismod_run([["init"], ["fit"], ["predict", "var"]])(rc)

    lines_read = db_file.open("r").readlines()
    assert len(lines_read) == 3


def test_recipe_not_found(dmdummy):
    """The input database won't be found."""
    with pytest.raises(Exception):
        rc = RecipeContext("bogus", dmdummy)
        dr.dismod_recipe([["fit"]], rc)

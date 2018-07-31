from argparse import Namespace
import random
import subprocess

import pytest

from cascade.executor.multiprocess import graph_do, NotEnoughResources, ChildProcessProblem


DONE = list()


class PopenObject:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        self.done = False
        self.returncode = 0

    def poll(self):
        assert not self.done
        dice = random.uniform(0, 1)
        if dice < 0.1:
            self.done = True
            DONE.append(self.args)
            return True
        else:
            return None


def popen_ish(*args, **kwargs):
    return PopenObject(args, kwargs)


def test_graph_do(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", popen_ish)

    to_do = dict()
    for i in range(10):
        to_do[i] = Namespace(memory=1, args=[str(i)])

    def run_next(completed):
        return {x: y for (x, y) in to_do.items() if x not in completed}

    global DONE
    DONE = list()
    with pytest.raises(NotEnoughResources):
        graph_do(run_next, 0.5, sleep_duration=0)

    DONE = list()
    graph_do(run_next, 1, sleep_duration=0)

    DONE = list()
    graph_do(run_next, 2, sleep_duration=0)
    assert len(DONE) == 10

    DONE = list()
    graph_do(run_next, 20, sleep_duration=0)
    assert len(DONE) == 10


def test_graph_live():
    to_do = dict()
    for i in range(5):
        to_do[i] = Namespace(memory=1, args=["dmchat", "2", "0", "0"])

    def run_next(completed):
        return {x: y for (x, y) in to_do.items() if x not in completed}

    graph_do(run_next, 2, sleep_duration=1)


def test_graph_die():
    to_do = dict()
    for i in range(5):
        to_do[i] = Namespace(memory=1, args=["dmchat", "1", "7", "0"])

    def run_next(completed):
        return {x: y for (x, y) in to_do.items() if x not in completed}

    with pytest.raises(ChildProcessProblem):
        graph_do(run_next, 2, sleep_duration=0)

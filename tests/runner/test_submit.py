import pytest

import cascade.runner.submit
from cascade.runner.submit import template_to_args, qsub


@pytest.mark.parametrize("template,args", [
    (dict(q="all.q"), ["-q", "all.q"]),
    (dict(v=False), ["-v", "FALSE"]),
    (dict(q="all.q", P="proj"), ["-q", "all.q", "-P", "proj"]),
    (dict(l=dict(a=1, b=2)), ["-l", "a=1,b=2"]),  # noqa: E741
    (dict(l=dict(), P="proj"), ["-P", "proj"]),  # noqa: E741
    (dict(l=dict(archive=True)), ["-l", "archive=TRUE"]),  # noqa: E741
])
def test_template_to_args_happy(template, args):
    assert template_to_args(template) == args


def test_qsub_execute_mock(monkeypatch):
    desired = ["file.sh", "arg1"]

    def whatsit(command, args, **_kwargs):
        if command != "which qsub":
            assert [command] + args == [
                "qsub", "-terse", "-q", "all.q", "file.sh", "arg1"]
            return "mvid"
        else:
            return"qsub"

    monkeypatch.setattr(cascade.runner.submit, "run_check", whatsit)
    template = dict(q="all.q")
    assert qsub(template, desired) == "mvid"

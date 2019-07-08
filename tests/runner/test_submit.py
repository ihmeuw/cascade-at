from pathlib import Path
from time import sleep, time

import pytest

import cascade.runner.submit
from cascade.executor.execution_context import application_config
from cascade.runner.status import qstat_short
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


@pytest.fixture
def qsub_template():
    settings = application_config()["GridEngine"]
    template = dict(
        l=dict(
            h_rt="00:05:00",
            m_mem_free="1G",
            fthread=1,
            archive=True,
        ),
        P=settings["project"],
        q=settings["queues"][0],
    )
    return template


STATECHART = dict(
    initial=dict(timeout=60, failure=True),
    engine=dict(timeout=600, failure=False),
    done=dict(timeout=0, failure=True),
)
"""
Only care about three states, the initial submission,
whether qstat has said it sees the file,
and done, whether that's out of qstat or that the
file exists.
"""


def check_complete(identify_job, check_done):
    """
    Submit a job and check that it ran.
    If the job never shows up in the queue, and
    it didn't run, that's a failure. If it shows up in
    the queue and goes over the timeout, we abandon it,
    because these are tests.

    Args:
        identify_job (function): True if it's this job.
        check_done (function): True if job is done.

    Returns:
        bool: False if the job ran over a timeout.
    """
    state = "initial"
    last = time()
    dead_to_me = {"deleted", "suspended"}
    while state != "done" and not check_done():
        my_jobs = qstat_short()
        this_job = [j for j in my_jobs if identify_job(j)]
        if len(this_job) == 1:
            print(this_job[0])
            if state == "initial":
                last = time()
                state = "engine"
            assert not (this_job[0].status & dead_to_me)
        elif len(this_job) == 0 and state == "engine":
            last = time()
            state = "done"
        else:
            assert len(this_job) < 2
        timeout = STATECHART[state]["timeout"]
        if time() - last > timeout:
            print(f"timeout for state {state}")
            assert STATECHART[state]["failure"]
            return False
        print(f"state {state}")
        if state != "done":
            sleep(15)
    return True


@pytest.mark.parametrize("queue_idx", [0, 1, 2])
def test_live_qsub(cluster, qsub_template, queue_idx):
    """Test the basic submission.
    This is basically all we will use from qsub.
    Note that it tests the project and queue.
    """
    qsub_template["b"] = "y"
    settings = application_config()["GridEngine"]
    queues = settings["queues"]
    if queue_idx < len(queues):
        qsub_template["q"] = queues[queue_idx]
    else:
        return
    job_name = "echo_test"
    qsub_template["N"] = job_name
    out_path = Path().cwd() / "live_qsub.out"
    if out_path.exists():
        out_path.unlink()
    qsub_template["o"] = out_path
    job_id = qsub(qsub_template, ["/bin/echo", "borlaug"])
    print(f"job_id {job_id}")

    def this_job(job):
        return job.job_id == job_id

    def check_done():
        return out_path.exists()

    ran_to_completion = check_complete(this_job, check_done)
    if ran_to_completion:
        if not out_path.exists():
            sleep(20)
        with out_path.open() as istream:
            contents = istream.read()
        assert "borlaug" in contents
        out_path.unlink()

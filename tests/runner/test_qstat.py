import xml.etree.ElementTree as ET

from cascade.runner.status import for_each_member, FlyWeightJob, job_states


def test_for_each_member_happy():
    tree = ET.parse('qstat_sample0.xml')
    structure = for_each_member(tree.getroot().find("djob_info"))
    job = FlyWeightJob(structure[0])
    print(job.status)
    print(job.name)
    print(job.job_id)
    print(job.task_cnt)
    assert job.task_cnt == 1
    task = job.tasks[0]
    assert int(task.number) == 1
    assert task.status == job.status
    assert not task.restarted
    assert len(task.hostname) > 0


def test_job_states_happy():
    states = job_states()
    assert len(states) == 13
    for v in states.values():
        assert isinstance(v, int)

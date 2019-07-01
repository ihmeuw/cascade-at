import xml.etree.ElementTree as ET
from datetime import datetime
from functools import lru_cache
from os import linesep
from textwrap import indent

from cascade.core.log import getLoggers
from cascade.executor.execution_context import application_config
from .grid_process import run_check

CODELOG, MATHLOG = getLoggers(__name__)


def qstat_request(effective_user=None, job_list=None):
    """
    Queries qstat with optional restrictions on jobs, to get XML output.
    It's important to use the job list as much as possible to restrict
    the job IDs queried.

    Args:
        effective_user (str): User ID of effective user.
        job_list (str): Job ids, job names, or wildcard expressions,
            as described in sge_types.
    Returns:
        (str|None): XML output from qstat.
    """
    args = ["-xml"]
    args.extend(["-u", str(effective_user)] if effective_user else [])
    if job_list:
        args.extend(["-j", str(job_list)])
    return run_check("qstat", args)


@lru_cache(maxsize=1)
def job_states():
    states = dict()
    for l in application_config()["GridEngine"]["job-states"].strip().split(linesep):
        tokens = l.split()
        states[tokens[1][1:].lower()] = int(tokens[2], 16)
    return states


def for_each_member(root):
    """Qstat XML has a very regular structure to define lists of
    dictionaries. This pulls that out to get a list of jobs
    where each job is a dictionary, and one member is the list
    of tasks contained."""
    # All lists contain multiple tags named element or grl.
    is_list = any(root.find(list_tag)
                  for list_tag in ["element", "grl", "job_list"])
    if is_list:
        child_list = list()
        for c in root:
            child_list.append(for_each_member(c))
        return child_list
    else:
        # Otherwise all the members are unique.
        child_dict = dict()
        for c in root:
            child_dict[c.tag] = for_each_member(c)

    if child_dict:
        return child_dict
    else:
        # If there were no members, then the information is in the text
        # of this element.
        return root.text


class FlyWeightJob:
    """Sits on top of the parsed XML to answer job questions.
    The `for_each_member` creates a reasonable Pythonic data structure. What's
    missing at that point is knowing what tag correspondes to what human
    information. We layer that here and will add what we need when we
    need it.
    """
    def __init__(self, job_jsonlike):
        """This is on top of the output of `for_each_member`."""
        self._json = job_jsonlike

    @property
    def status(self):
        """Set of strings like: idle, running, as a set.
        This can be in more than one state at a time, such as
        ``{"queued", "waiting"}``, which we know as qw.
        """
        if self.task_cnt < 1:
            return {"idle"}
        elif self.task_cnt > 1:
            raise ValueError(f"This job has {self.task_cnt} tasks")
        else:
            return FlyWeightTask(self._json["JB_ja_tasks"][0]).status

    @property
    def tasks(self):
        """FlyWeightTasks for tasks in the job. Can be none."""
        return [FlyWeightTask(t) for t in self._json["JB_ja_tasks"]]

    @property
    def name(self):
        """As given by the ``-N`` qsub option."""
        return self._json["JB_job_name"]

    @property
    def job_id(self):
        """The job ID, as in 2349272."""
        return self._json["JB_job_number"]

    @property
    def task_cnt(self):
        """How many tasks are associated with this job.
        Jobs contain tasks, and it's the tasks that run, have statuses,
        and have CPU times.
        """
        if "JB_ja_tasks" in self._json:
            return len(self._json["JB_ja_tasks"])
        else:
            return 0


class FlyWeightTask:
    def __init__(self, task_jsonlike):
        """Pass into this each ``JB_ja_tasks`` member of the job."""
        self._task = task_jsonlike

    @property
    def number(self):
        """Tasks within a job are numbered from 1."""
        return self._task["JAT_task_number"]

    @property
    def status(self):
        """Status is a set of strings."""
        raw_state = int(self._task["JAT_status"])
        state = set()
        state_categories = job_states()
        for state_name, mask in state_categories.items():
            if mask & raw_state != 0:
                state.add(state_name)
        if not state:
            state = {"idle"}
        return state

    @property
    def restarted(self):
        """Bool: Whether this task did restart."""
        return self._task["JAT_job_restarted"] != "0"

    @property
    def hostname(self):
        """Hostname where this task will run, is running, or has run."""
        places = self._task["JAT_granted_destin_identifier_list"]
        if len(places) > 0:
            return places[0]["JG_qhostname"]
        else:
            return None


def qstat(effective_user=None, job_list=None):
    """Get status of all jobs in the job_list belonging to the given
    user::

        import getpass
        user = getpass.getuser()
        job_info = qstat(user, "dm_38044_*")

    Args:
        effective_user (str): The user ID.
        job_list (str): Can be model version IDs, or a job name,
            or a job name with a wildcard. See ``man sge_types``.
    Returns:
        List[FlyWeightJob]: Information about the jobs.
    """
    # Breaks qstat into three layers. Call it to get xml,
    #     parse XML into Python lists and dicts. Then put flyweight
    #     classes on those lists and dicts.
    job_list = job_list if job_list else "*"
    xml = qstat_request(effective_user, job_list)
    if xml is None:
        return list()
    CODELOG.debug(f"xml is {len(xml)} characters {xml[:400]}")
    # The xml uses a tag with a space in it. Here's an ugly fix.
    xml = xml.replace("job args>", "job_args>")
    try:
        parsed = ET.fromstring(xml)
    except ET.ParseError as pe:
        line, column = pe.position
        lines = xml.splitlines()
        bad_line = lines[line][:column] + "*" + lines[line][column:]
        lines[line] = bad_line
        snippet = lines[max(line - 3, 0):min(line + 4, len(lines))]
        formatted = indent(linesep.join(snippet), " " * 4)
        CODELOG.exception(f"Could not parse qstat output {pe}:\n{formatted}")
        raise RuntimeError(f"Could not parse qstat output", bad_line)
    job_data = for_each_member(parsed.find("djob_info"))
    return [FlyWeightJob(jd) for jd in job_data]


LETTER_CODE_TO_STATE = dict(
    q="queued",
    r="running",
    d="deleted",
    t="transfering",
    h="held",
    m="migrating",
    s="suspended",
    w="waiting",
    e="exiting",
)


class MiteWeightJob:
    """Like the FlyWeightJob, this represents a Job. This one
    includes everything in the simplified version of qstat.
    """
    def __init__(self, job_jsonlike):
        """This is on top of the output of `for_each_member`."""
        self._json = job_jsonlike

    @property
    def status(self):
        """Set of strings like: idle, running, as a set.
        This can be in more than one state at a time, such as
        ``{"queued", "waiting"}``, which we know as qw.
        """
        state = set()
        for letter in self._json["state"]:
            if letter in LETTER_CODE_TO_STATE:
                state.add(LETTER_CODE_TO_STATE[letter])
        if not state:
            state.add("idle")
        return state

    @property
    def tasks(self):
        """FlyWeightTasks for tasks in the job. Can be none."""
        return list()

    @property
    def name(self):
        """As given by the ``-N`` qsub option."""
        return self._json["JB_name"]

    @property
    def owner(self):
        return self._json["JB_owner"]

    @property
    def priority(self):
        return float(self._json["JAT_prio"])

    @property
    def start_time(self):
        if "JAT_start_time" in self._json:
            time_string = self._json["JAT_start_time"]
            return datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            return None

    @property
    def submission_time(self):
        if "JB_submission_time" in self._json:
            time_string = self._json["JB_submission_time"]
            return datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            return None

    @property
    def job_id(self):
        """The job ID, as in 2349272."""
        return self._json["JB_job_number"]

    @property
    def task_cnt(self):
        """How many tasks are associated with this job.
        Jobs contain tasks, and it's the tasks that run, have statuses,
        and have CPU times.
        """
        return 0


def qstat_short(effective_user=None):
    """
    Calling qstat without -j gets a much smaller result that just has
    job information.
    """
    xml = qstat_request(effective_user)
    if xml is None:
        return list()
    parsed = ET.fromstring(xml)
    job_list = list()
    for sub_tag in ["queue_info", "job_info"]:
        sub_root = parsed.find(sub_tag)
        if sub_root:
            job_list.extend(for_each_member(sub_root))
    return [MiteWeightJob(jd) for jd in job_list]

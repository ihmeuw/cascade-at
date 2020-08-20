import os
import getpass
from typing import Optional

from cascade_at.core.db import swarm
from cascade_at.core.log import get_loggers
from cascade_at.cascade.cascade_operations import _CascadeOperation

LOG = get_loggers(__name__)

Workflow = swarm.workflow.workflow.Workflow
BashTask = swarm.workflow.bash_task.BashTask
ExecutorParameters = swarm.executors.base.ExecutorParameters

# This is just here to skip the ModuleProxy so that I can
# develop better in PyCharm
from jobmon.client.api import Tool, ExecutorParameters


class JobmonConstants:
    EXECUTOR = "SGEExecutor"
    PROJECT = "proj_dismod_at"


def bash_task_from_cascade_operation(co: _CascadeOperation, tool: Tool) -> BashTask:
    """
    Create a Jobmon bash task from a cascade operation (co for short).
    """
    template = co.get_task_template(tool)
    template.create_task(
        name=co.name,
        max_attempts=3,
        executor_parameters=ExecutorParameters(
            max_runtime_seconds=co.executor_parameters['max_runtime_seconds'],
            j_resource=co.j_resource,
            m_mem_free=co.executor_parameters['m_mem_free'],
            num_cores=co.executor_parameters['num_cores'],
            resource_scales=co.executor_parameters['resource_scales']
        ),
        **kwargs
    )


def jobmon_workflow_from_cascade_command(cc, context, addl_workflow_args: Optional[str] = None):
    """
    Create a jobmon workflow from a cascade command (cc for short)

    Parameters
    ----------
    cc
        The cascade command
    context
    addl_workflow_args
        Additional workflow args to add on
    """
    error_dir = context.log_dir / 'errors'
    output_dir = context.log_dir / 'output'

    for folder in [error_dir, output_dir]:
        os.makedirs(folder, exist_ok=True)

    workflow_args = f'dismod-at-{cc.model_version_id}'
    if addl_workflow_args:
        workflow_args += f'-{addl_workflow_args}'

    tool = Tool.create_tool(name="dismod-at")
    wf = tool.create_workflow(name=workflow_args)
    wf.set_executor(
        executor_class=JobmonConstants.EXECUTOR,
        project=JobmonConstants.PROJECT
    )

    # wf = Workflow(
    #     workflow_args=workflow_args,
    #     project='proj_dismod_at',
    #     stderr=str(error_dir),
    #     stdout=str(output_dir),
    #     working_dir=str(context.model_dir),
    #     seconds_until_timeout=60*60*24*5,
    #     resume=True
    # )
    bash_tasks = dict()
    for command, co in cc.task_dict.items():
        task = bash_task_from_cascade_operation(co=co, tool=tool)
        for uc in co.upstream_commands:
            task.add_upstream(bash_tasks.get(uc))
        bash_tasks.update({
            command: task
        })
    wf.add_tasks(list(bash_tasks.values()))
    return wf

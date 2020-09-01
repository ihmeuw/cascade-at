import os
from typing import Optional

from cascade_at.cascade.cascade_operations import _CascadeOperation
from cascade_at.core.db import api, task, task_template, strategies
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)

ExecutorParameters = api.ExecutorParameters
Tool = api.Tool
Task = task.Task
Template = task_template.TaskTemplate
SGEExecutor = strategies.sge


class JobmonConstants:
    EXECUTOR = "SGEExecutor"
    PROJECT = "proj_dismod_at"


def task_template_from_cascade_operation(co: _CascadeOperation, tool: Tool) -> TaskTemplate:
    return tool.get_task_template(
        template_name=co._script(),
        command_template="{script} " + f"{co.arg_list.template}",
        node_args=co.arg_list.node_args,
        task_args=co.arg_list.task_args
    )


def task_from_cascade_operation(co: _CascadeOperation, tool: Tool) -> Task:
    """
    Create a Jobmon task from a cascade operation (co for short).
    """
    template = task_template_from_cascade_operation(co, tool)
    task = template.create_task(
        name=co.name,
        max_attempts=3,
        executor_parameters=ExecutorParameters(
            max_runtime_seconds=co.executor_parameters['max_runtime_seconds'],
            j_resource=co.j_resource,
            m_mem_free=co.executor_parameters['m_mem_free'],
            num_cores=co.executor_parameters['num_cores'],
            resource_scales=co.executor_parameters['resource_scales']
        ),
        script=co._script(),
        **co.template_kwargs
    )
    return task


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
        project=JobmonConstants.PROJECT,
        stderr=str(error_dir),
        stdout=str(output_dir),
        working_dir=str(context.model_dir),
    )
    bash_tasks = dict()
    for command, co in cc.task_dict.items():
        task = task_from_cascade_operation(co=co, tool=tool)
        for uc in co.upstream_commands:
            task.add_upstream(bash_tasks.get(uc))
        bash_tasks.update({
            command: task
        })
    wf.add_tasks(list(bash_tasks.values()))
    return wf

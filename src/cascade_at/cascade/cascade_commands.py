"""
================
Cascade Commands
================

Sequences of cascade operations that work together to create a cascade command
that will run the whole cascade (or a drill -- which is a version of the cascade).
"""
from cascade_at.core.log import get_loggers
from cascade_at.cascade.cascade_stacks import single_fit


LOG = get_loggers(__name__)


class _CascadeCommand:
    def __init__(self):
        self.task_dict = {}

    def add_task(self, cascade_operation):
        self.task_dict.update({
            cascade_operation.command: cascade_operation
        })

    def get_commands(self):
        """
        Gets a list of commands in sequence so that you can run
        them without using jobmon.
        """
        return list(self.task_dict.keys())


class Drill(_CascadeCommand):
    """
    Runs a drill!
    """
    def __init__(self, model_version_id,
                 drill_parent_location_id, drill_sex):
        super().__init__()
        tasks = single_fit(
            model_version_id=model_version_id,
            location_id=drill_parent_location_id,
            sex_id=drill_sex,
        )
        for t in tasks:
            self.add_task(t)


class TraditionalCascade(_CascadeCommand):
    """
    Runs the traditional cascade.
    """
    def __init__(self, model_version_id):
        super().__init__()

        raise NotImplementedError

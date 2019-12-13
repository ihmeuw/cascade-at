"""
Sequences of cascade operations that work together to create a cascade command
that will run the whole cascade (or a drill -- which is a version of the cascade).
"""
from cascade_at.core.log import get_loggers
from cascade_at.cascade.cascade_operations import CASCADE_OPERATIONS

LOG = get_loggers(__name__)


class CascadeCommand:
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
        :return:
        """
        return list(self.task_dict.keys())


class Drill(CascadeCommand):
    """
    Runs a drill!
    """
    def __init__(self, model_version_id,
                 drill_parent_location_id, drill_sex):
        super().__init__()
        self.model_version_id = model_version_id
        self.drill_parent_location_id = drill_parent_location_id
        self.drill_sex = drill_sex

        self.add_task(CASCADE_OPERATIONS['configure_inputs'](
            model_version_id=self.model_version_id,
            drill_parent_location_id=self.drill_parent_location_id
        ))
        self.add_task(CASCADE_OPERATIONS['fit_both'](
            model_version_id=self.model_version_id,
            parent_location_id=self.drill_parent_location_id,
            sex_id=self.drill_sex,
            upstream_commands=self.get_commands()
        ))
        self.add_task(CASCADE_OPERATIONS['format_upload'](
            model_version_id=self.model_version_id,
            parent_location_id=self.drill_parent_location_id,
            sex_id=self.drill_sex,
            upstream_commands=self.get_commands()
        ))


class TraditionalCascade(CascadeCommand):
    """
    Runs the traditional cascade.
    """
    def __init__(self, model_version_id):
        super().__init__()
        self.model_version_id = model_version_id
        raise NotImplementedError("Cascade is not implemented yet for Cascade-AT.")


CASCADE_COMMANDS = {
    'drill': Drill,
    'cascade': TraditionalCascade
}

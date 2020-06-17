"""
================
Cascade Commands
================

Sequences of cascade operations that work together to create a cascade command
that will run the whole cascade (or a drill -- which is a version of the cascade).
"""
from cascade_at.core.log import get_loggers
from cascade_at.cascade.cascade_stacks import single_fit
from cascade_at.cascade.cascade_dags import make_cascade_dag
from cascade_at.inputs.locations import LocationDAG
from cascade_at.inputs.utilities.gbd_ids import SEX_NAME_TO_ID, CascadeConstants


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
    def __init__(self, model_version_id: int,
                 drill_parent_location_id: int, drill_sex: int):
        super().__init__()

        self.model_version_id = model_version_id
        self.drill_parent_id = drill_parent_location_id
        self.drill_sex = drill_sex

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
    def __init__(self, model_version_id: int, split_sex: bool,
                 dag: LocationDAG, n_sim: int):
        super().__init__()
        self.model_version_id = model_version_id

        tasks = make_cascade_dag(
            model_version_id=model_version_id,
            dag=dag,
            location_start=CascadeConstants.GLOBAL_LOCATION_ID,
            sex_start=SEX_NAME_TO_ID['Both'],
            split_sex=split_sex,
            n_sim=n_sim, n_pool=10
        )

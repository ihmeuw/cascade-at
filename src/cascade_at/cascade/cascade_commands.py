"""
================
Cascade Commands
================

Sequences of cascade operations that work together to create a cascade command
that will run the whole cascade (or a drill -- which is a version of the cascade).
"""
from typing import Optional, List

from cascade_at.core.log import get_loggers
from cascade_at.cascade.cascade_stacks import single_fit_with_uncertainty
from cascade_at.cascade.cascade_dags import make_cascade_dag
from cascade_at.cascade.cascade_operations import _CascadeOperation
from cascade_at.inputs.locations import LocationDAG
from cascade_at.inputs.utilities.gbd_ids import SEX_NAME_TO_ID, CascadeConstants


LOG = get_loggers(__name__)


class _CascadeCommand:
    """
    Base class for a cascade command.
    """
    def __init__(self):
        """
        Initializes a task dictionary. All tasks added to this command
        in the form of cascade operations are added to the dictionary.

        Attributes
        ----------
        self.task_dict
            A dictionary of cascade operations, keyed by the command
            for that operation. This is so that we can look up the
            task later by the exact command.
        """
        self.task_dict = {}

    def add_task(self, cascade_operation: _CascadeOperation) -> None:
        """
        Adds a cascade operation to the task dictionary.

        Parameters
        ----------
        cascade_operation
            A cascade operation to add to the command dictionary
        """
        self.task_dict.update({
            cascade_operation.command: cascade_operation
        })

    def get_commands(self) -> List[str]:
        """
        Gets a list of commands in sequence so that you can run
        them without using jobmon.

        Returns
        -------
        Returns a list of commands that you can run on the command-line.
        """
        return list(self.task_dict.keys())


class Drill(_CascadeCommand):
    def __init__(self, model_version_id: int,
                 drill_parent_location_id: int, drill_sex: int,
                 n_sim: int, n_pool: int = 10,
                 json_file: Optional[str] = ''):
        """
        A cascade command that runs a drill model, meaning
        that it runs one Dismod-AT model with a parent
        plus its children.

        Parameters
        ----------
        model_version_id
            The model version ID to create the drill for
        drill_parent_location_id
            The parent location ID to start the drill from
        drill_sex
            Which sex to drill for
        n_sim
            The number of simulations to do to get uncertainty at the leaf nodes
        n_pool
            The number of threads to create in a multiprocessing pool.
            If this is 1, then it will not do multiprocessing.
        json_file
            Pass this argument do configure_inputs
        """
        super().__init__()

        self.model_version_id = model_version_id
        self.drill_parent_id = drill_parent_location_id
        self.drill_sex = drill_sex

        tasks = single_fit_with_uncertainty(
            model_version_id=model_version_id,
            location_id=drill_parent_location_id,
            sex_id=drill_sex,
            n_sim=n_sim,
            n_pool=n_pool,
            json_file=json_file,
        )
        for t in tasks:
            self.add_task(t)


class TraditionalCascade(_CascadeCommand):
    def __init__(self, model_version_id: int, split_sex: bool,
                 dag: LocationDAG, n_sim: int, n_pool: int = 10,
                 location_start: Optional[int] = None,
                 sex: Optional[int] = None, skip_configure: bool = False,
                 json_file: Optional[str] = ''):
        """
        Runs the "traditional" dismod cascade. The traditional cascade
        as implemented here runs fit fixed all the way to the leaf nodes of
        the cascade to save time (rather than fit both).
        To get posterior to prior it uses the coefficient of variation
        to get the variance of the posterior that becomes the prior
        at the next level. At the leaf nodes to get final posteriors,
        it does sample asymptotic. If sample asymptotic fails due to bad
        constraints it does sample simulate instead.

        Parameters
        ----------
        model_version_id
            The model version ID
        split_sex
            Whether or not to split sex
        dag
            A location dag that specifies the structure of the cascade hierarchy
        n_sim
            The number of simulations to do to get uncertainty at the leaf nodes
        n_pool
            The number of threads to create in a multiprocessing pool.
            If this is 1, then it will not do multiprocessing.
        location_start
            Which location to start the cascade from (typically 1 = Global)
        sex
            Which sex to run the cascade for (if it's 3 = Both, then it will
            split sex, if it's 1 or 2, then it will only run it for that sex.
        skip_configure
            Use this option to skip the initial inputs pulling; should only
            be used in debugging cases by developers.
        json_file
            Pass this argument do configure_inputs
        """

        super().__init__()
        self.model_version_id = model_version_id

        if sex is None:
            sex = SEX_NAME_TO_ID['Both']
        if location_start is None:
            location_start = CascadeConstants.GLOBAL_LOCATION_ID

        tasks = make_cascade_dag(
            model_version_id=model_version_id,
            dag=dag,
            location_start=location_start,
            sex_start=sex,
            split_sex=split_sex,
            n_sim=n_sim,
            n_pool=n_pool,
            skip_configure=skip_configure,
            json_file=json_file
        )
        for t in tasks:
            self.add_task(t)

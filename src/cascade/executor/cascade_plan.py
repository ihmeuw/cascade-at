"""
Specification for a whole cascade.
"""

import networkx as nx

from cascade.core import getLoggers
from cascade.core.parameters import ParameterProperty
from cascade.input_data import InputDataError
from cascade.input_data.configuration.builder import policies_from_settings
from cascade.input_data.db.locations import (
    location_id_from_start_and_finish
)

CODELOG, MATHLOG = getLoggers(__name__)


class EstimationParameters:
    def __init__(self, settings, policies, children,
                 parent_location_id, grandparent_location_id, sex_id):

        self.parent_location_id = parent_location_id
        self.sex_id = sex_id
        self.data_access = ParameterProperty()
        """These decide which data to get."""

        self.run = ParameterProperty()
        """These affect how the program runs but not its results."""

        self.grandparent_location_id = grandparent_location_id
        """Can be null at top of drill, even when not global location."""

        self.children = children
        self.settings = settings
        self.policies = policies


class CascadePlan:
    """
    Clients are the EpiViz-AT runner and an interactive user for configuration.
    Collaborators are tools to build a DismodFile object and process its output.
    Responsible for strategies

     * to build a model, and
     * to bootstrap the next model.

    This knows the hierarchy of Dismod-AT models that must be fit,
    simulated, predicted, and aggregated. Each model in the hierarchy
    has a unique identifier of the form ``(location_id, index)`` where
    the index would be 0 for the initial fit and increment for each
    simulation, for instance.

    Each location is at a level in the hierarchy. Most of the specification
    depends on the level of the location within the hierarchy.
    """

    def __init__(self, settings):
        """
        There are two kinds of clients, EpiViz-AT and interactive users.
        EpiViz-AT uses its settings to parameterize this class. Both
        clients rely on default policies.
        """
        self._locations = None
        self._task_graph = None
        self._settings = settings
        self._args = None

    @property
    def cascade_jobs(self):
        return nx.lexicographical_topological_sort(self._task_graph)

    def cascade_job(self, cascade_job_id):
        """Given settings and an identifier for this job, return a local
        version of settings that describes this particular job.

        Adds ``settings.model.parent_location_id``,
        ``settings.model.grandparent_location_id``,
        and ``settings.model.children``.
        There is a grandparent location only if there is a grandparent task,
        so a drill starting halfway will not have a grandparent location.
        There are child locations for the last task though.
        """
        parent_task = list(self._task_graph.in_edges(cascade_job_id))
        if parent_task:
            # [only edge][(edge start, edge finish)][(location, index)]
            grandparent_location_id = self._location_of_cascade_job(parent_task[0][0])
        else:
            grandparent_location_id = None

        print(f"settings {type(self._settings)} {self._settings.policies}")
        parent_location_id = self._location_of_cascade_job(cascade_job_id)

        policies = policies_from_settings(self._settings)
        local_settings = EstimationParameters(
            settings=self._settings,
            policies=policies,
            children=list(sorted(self._locations.successors(parent_location_id))),
            parent_location_id=parent_location_id,
            grandparent_location_id=grandparent_location_id,
            sex_id=self._settings.model.drill_sex,
        )
        local_settings.data_access = dict(
            gbd_round_id=self._settings.gbd_round_id,
            modelable_entity_id=self._settings.model.modelable_entity_id,
            model_version_id=self._settings.model.model_version_id,
            settings_file=self._args.settings_file,
            bundle_file=self._args.bundle_file,
            bundle_study_covariates_file=self._args.bundle_study_covariates_file,
            tier=2 if self._args.skip_cache else 3,
            age_group_set_id=policies["age_group_set_id"],
            with_hiv=policies["with_hiv"]
        )
        local_settings.run = dict(
            no_upload=self._args.no_upload,
            db_only=self._args.db_only,
            num_processes=self._args.num_processes,
            pdb=self._args.pdb,
        )
        return "estimate_location", local_settings

    @classmethod
    def from_epiviz_configuration(cls, locations, settings, args):
        """

        Args:
            locations (nx.Graph): A graph of locations in a hierarchy.
            settings (Configuration): The EpiViz-AT Form (in form.py)
            args (argparse.Namespace): Parsed arguments.

        """
        plan = cls(settings)
        plan._locations = locations
        plan._args = args
        if hasattr(settings.model, "drill_location_start") and \
                settings.model.drill_location_start and settings.model.drill_location_end:
            try:
                drill = location_id_from_start_and_finish(
                    plan._locations, settings.model.drill_location_start, settings.model.drill_location_end)
            except ValueError as ve:
                raise InputDataError(f"Location parameter is wrong in settings.") from ve
        else:
            MATHLOG.error(f"Looking for drill start and finish and cannot find "
                          f"drill location start and end.")
            raise InputDataError(f"Missing drill location start and end.")
        MATHLOG.info(f"drill nodes {', '.join(str(d) for d in drill)}")
        tasks = [(drill_location, 0) for drill_location in drill]
        task_pairs = list(zip(tasks[:-1], tasks[1:]))
        plan._task_graph = nx.DiGraph()
        plan._task_graph.add_nodes_from(tasks)
        plan._task_graph.add_edges_from(task_pairs)
        # Add a custom graph attribute to record the tree root.
        plan._task_graph.graph["root"] = tasks[0]
        return plan

    def _location_of_cascade_job(self, cascade_job_id):
        return cascade_job_id[0]

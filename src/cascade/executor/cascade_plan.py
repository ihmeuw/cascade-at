"""
Specification for a whole cascade.
"""
from copy import deepcopy
import networkx as nx

from cascade.core import getLoggers
from cascade.input_data import InputDataError
from cascade.input_data.db.locations import (
    location_id_from_location_and_level, location_id_from_start_and_finish
)


CODELOG, MATHLOG = getLoggers(__name__)


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
        local_settings = deepcopy(self._settings)
        parent_location_id = cascade_job_id[0]
        local_settings.model.parent_location_id = parent_location_id
        parent_task = list(self._task_graph.in_edges(cascade_job_id))
        if parent_task:
            # only edge, starting graph node, (location, index)
            local_settings.model.grandparent_location_id = parent_task[0][0][0]
        else:
            local_settings.model.grandparent_location_id = None
        local_settings.model.children = set(self._locations.successors(parent_location_id))
        return "estimate_location", local_settings

    @classmethod
    def from_epiviz_configuration(cls, locations, settings):
        plan = cls(settings)
        plan._locations = locations
        if hasattr(settings.model, "drill_location_start") and \
                settings.model.drill_location_start and settings.model.drill_location_end:
            try:
                drill = location_id_from_start_and_finish(
                    plan._locations, settings.model.drill_location_start, settings.model.drill_location_end)
            except ValueError as ve:
                raise InputDataError(f"Location parameter is wrong in settings.") from ve
        elif settings.model.drill_location:
            starting_level = settings.model.split_sex
            end_location = settings.model.drill_location
            drill = location_id_from_location_and_level(plan._locations, end_location, starting_level)
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

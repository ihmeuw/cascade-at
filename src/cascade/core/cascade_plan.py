"""
Specification for a whole cascade.
"""
import networkx as nx

from cascade.input_data.db.locations import location_hierarchy, location_id_from_location_and_level
from cascade.core import getLoggers
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

    def __init__(self):
        """
        There are two kinds of clients, EpiViz-AT and interactive users.
        EpiViz-AT uses its settings to parameterize this class. Both
        clients rely on default policies.
        """
        self.locations = None
        self.task_graph = None

    @property
    def tasks(self):
        return nx.lexicographical_topological_sort(self.task_graph)

    @classmethod
    def from_epiviz_configuration(cls, execution_context, settings):
        plan = cls()
        plan.policies = settings.policies

        plan.locations = location_hierarchy(execution_context)
        starting_level = settings.model.split_sex
        end_location = settings.model.drill_location
        drill = location_id_from_location_and_level(execution_context, end_location, starting_level)
        MATHLOG.info(f"drill nodes {', '.join(str(d) for d in drill)}")
        if len(drill) > 1:
            drill_start = drill
            drill = drill[-1:]
            MATHLOG.warning(f"Reducing drill nodes to {', '.join(str(d) for d in drill)} from {drill_start}")
        tasks = [(drill_location, 0) for drill_location in drill]
        task_pairs = list(zip(tasks[:-1], tasks[1:]))
        plan.task_graph = nx.DiGraph()
        plan.task_graph.add_nodes_from(tasks)
        plan.task_graph.add_edges_from(task_pairs)
        # Add a custom graph attribute to record the tree root.
        plan.task_graph.graph["root"] = tasks[0]
        return plan

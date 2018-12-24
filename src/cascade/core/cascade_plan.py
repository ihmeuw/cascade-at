"""
Specification for a whole cascade.
"""
from cascade.input_data.db.locations import get_location_hierarchy_from_gbd, location_id_from_location_and_level
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
        # The type for this is defined by the ``hierarchy.trees.tree`` class.
        self.location_hierarchy = None
        self.locations_for_run = None

    def from_epiviz_configuration(self, execution_context, configuration):
        settings = configuration.settings
        self.policies = settings.policies

        self.location_hierarchy = get_location_hierarchy_from_gbd(execution_context)
        starting_level = settings.model.split_sex
        end_location = settings.model.drill_location
        location_id_from_location_and_level(execution_context, end_location, starting_level)

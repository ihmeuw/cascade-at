"""
Specification for a whole cascade.
"""
import networkx as nx

from cascade.core import getLoggers
from cascade.core.parameters import ParameterProperty, _ParameterHierarchy
from cascade.input_data import InputDataError
from cascade.input_data.configuration.builder import policies_from_settings
from cascade.input_data.db.locations import (
    location_id_from_start_and_finish
)

CODELOG, MATHLOG = getLoggers(__name__)


class EstimationParameters:
    def __init__(self, settings, policies, children,
                 parent_location_id, grandparent_location_id, sexes, number_of_fixed_effect_samples,
                 model_options):

        self.parent_location_id = parent_location_id
        self.sexes = sexes
        self.data_access = ParameterProperty()
        """These decide which data to get."""

        self.run = ParameterProperty()
        """These affect how the program runs but not its results."""

        self.grandparent_location_id = grandparent_location_id
        """Can be null at top of drill, even when not global location."""

        self.model_options = model_options

        self.children = children
        self.settings = settings
        self.policies = policies
        self.number_of_fixed_effect_samples = number_of_fixed_effect_samples


def make_model_options(locations, parent_location_id, ev_settings):
    bound_random = get_bound_random_this_location(locations, parent_location_id, ev_settings)

    model_options = _ParameterHierarchy(**dict(
            bound_random=bound_random,
        ))
    return model_options


def get_bound_random_this_location(locations, parent_location_id, ev_settings):
    # Set the bounds throughout the location hierarchy.
    # hasattr is right here because any unset ancestor makes the parent unset.
    # and one  of the child forms can have an unset location or value.
    if hasattr(ev_settings, "re_bound_location"):
        add_bound_random_to_location_properties(ev_settings.re_bound_location, locations)
    else:
        CODELOG.debug("No re_bound_location in settings.")

    # Get the global value, if it exists.
    if not ev_settings.model.is_field_unset("bound_random"):
        bound_random = ev_settings.model.bound_random
    else:
        bound_random = None
    CODELOG.debug(f"Setting bound_random's default to {bound_random}")

    # Search up the location hierarchy to see if an ancestor has a value.
    this_and_ancestors = nx.ancestors(locations, parent_location_id) | {
    parent_location_id}
    to_top = list(nx.topological_sort(nx.subgraph(locations, this_and_ancestors)))
    to_top.reverse()
    for check_bounds in to_top:
        if "bound_random" in locations.node[check_bounds]:
            CODELOG.debug(f"Found bound random in location {check_bounds}")
            bound_random = locations.node[check_bounds]["bound_random"]
            break
    return bound_random


def add_bound_random_to_location_properties(re_bound_location, locations):
    for bounds_form in re_bound_location:
        if not bounds_form.is_field_unset("value"):
            value = bounds_form.value
        else:
            value = None  # This turns off bound random option.

        if not bounds_form.is_field_unset("location"):
            CODELOG.debug(f"setting {bounds_form.location} to {value}")
            locations.node[bounds_form.location]["bound_random"] = value
        else:
            CODELOG.debug(f"setting root to {value}")
            locations.node[locations.graph["root"]]["bound_random"] = value


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
        if parent_task and self._job_kind(parent_task[0][0]) == "estimate_location":
            # [only edge][(edge start, edge finish)][(location, index)]
            grandparent_location_id = self._location_of_cascade_job(parent_task[0][0])
        else:
            grandparent_location_id = None

        parent_location_id = self._location_of_cascade_job(cascade_job_id)
        if self._settings.model.is_field_unset("drill_sex"):
            # An unset drill sex gets all data.
            sexes = [1, 2, 3]
        else:
            # Setting to male or female pulls in "both."
            sexes = [self._settings.model.drill_sex, 3]

        policies = policies_from_settings(self._settings)
        model_options = make_model_options(self._locations, parent_location_id, self._settings)
        if self._args.num_samples:
            sample_cnt = self._args.num_samples
        else:
            sample_cnt = policies["number_of_fixed_effect_samples"]

        local_settings = EstimationParameters(
            settings=self._settings,
            policies=policies,
            children=list(sorted(self._locations.successors(parent_location_id))),
            parent_location_id=parent_location_id,
            grandparent_location_id=grandparent_location_id,
            # This is a list of [1], [3], [1,3], [2,3], [1,2,3], not [1,2].
            sexes=sexes,
            number_of_fixed_effect_samples=sample_cnt,
            model_options=model_options,
        )
        local_settings.data_access = _ParameterHierarchy(**dict(
            gbd_round_id=self._settings.gbd_round_id,
            modelable_entity_id=self._settings.model.modelable_entity_id,
            model_version_id=self._settings.model.model_version_id,
            settings_file=self._args.settings_file,
            bundle_file=self._args.bundle_file,
            bundle_id=self._settings.model.bundle_id,
            bundle_study_covariates_file=self._args.bundle_study_covariates_file,
            tier=2 if self._args.skip_cache else 3,
            age_group_set_id=policies["age_group_set_id"],
            with_hiv=policies["with_hiv"],
            cod_version=self._settings.csmr_cod_output_version_id,
            location_set_version_id=self._settings.location_set_version_id,
            add_csmr_cause=self._settings.model.add_csmr_cause,
        ))
        local_settings.run = _ParameterHierarchy(**dict(
            no_upload=self._args.no_upload,
            db_only=self._args.db_only,
            num_processes=self._args.num_processes,
            pdb=self._args.pdb,
        ))
        return self._job_kind(cascade_job_id), local_settings

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
        if not settings.model.is_field_unset("drill") and settings.model.drill == "drill":
            if not settings.model.is_field_unset("drill_location_start"):
                drill_start = settings.model.drill_location_start
            else:
                drill_start = None
            if not settings.model.is_field_unset("drill_location_end"):
                drill_end = settings.model.drill_location_end
            else:
                raise InputDataError(f"Set to drill but drill location end not set")
            try:
                drill = location_id_from_start_and_finish(plan._locations, drill_start, drill_end)
            except ValueError as ve:
                raise InputDataError(f"Location parameter is wrong in settings.") from ve
        else:
            MATHLOG.error(f"Must be set to 'drill'")
            raise InputDataError(f"Must be set to 'drill'")
        MATHLOG.info(f"drill nodes {', '.join(str(d) for d in drill)}")
        drill = list(drill)
        if args.skip_cache:
            setup_task = []
        else:
            setup_task = [(drill[0], "bundle_setup")]
        tasks = setup_task + [(drill_location, 0) for drill_location in drill]
        task_pairs = list(zip(tasks[:-1], tasks[1:]))
        plan._task_graph = nx.DiGraph()
        plan._task_graph.add_nodes_from(tasks)
        plan._task_graph.add_edges_from(task_pairs)
        # Add a custom graph attribute to record the tree root.
        plan._task_graph.graph["root"] = tasks[0]
        return plan

    def _job_kind(self, cascade_job_id):
        if cascade_job_id[1] == "bundle_setup":
            return "bundle_setup"
        else:
            return "estimate_location"

    def _location_of_cascade_job(self, cascade_job_id):
        return cascade_job_id[0]

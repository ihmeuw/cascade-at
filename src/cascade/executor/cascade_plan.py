"""
Specification for what parameters are used at what location within
the Cascade.
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
    location_tree = locations.copy()
    if hasattr(ev_settings, "re_bound_location"):
        add_bound_random_to_location_properties(ev_settings.re_bound_location, location_tree)
    else:
        CODELOG.debug("No re_bound_location in settings.")

    # Get the global value, if it exists.
    if not ev_settings.model.is_field_unset("bound_random"):
        bound_random = ev_settings.model.bound_random
    else:
        bound_random = None
    CODELOG.debug(f"Setting bound_random's default to {bound_random}")

    # Search up the location hierarchy to see if an ancestor has a value.
    this_and_ancestors = nx.ancestors(location_tree, parent_location_id) | {parent_location_id}
    to_top = list(nx.topological_sort(nx.subgraph(location_tree, this_and_ancestors)))
    to_top.reverse()
    for check_bounds in to_top:
        if "bound_random" in location_tree.node[check_bounds]:
            CODELOG.debug(f"Found bound random in location {check_bounds}")
            bound_random = location_tree.node[check_bounds]["bound_random"]
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


class RecipeIdentifier:
    """This is a tuple that identifies a recipe within the graph of recipes.
    A Recipe is the set of steps executed for a single location, but it is
    also the set of steps run at the start or end of a whole model. For
    instance, aggregation is a recipe.

    1. Location ID, which may be 0, to indicate that this task is
       associated with no location, or all locations.
    2. A string identifier for the set of tasks at this location.

    Args:
        location_id (int): Location identifier from GBD.
        recipe (str): Identifies a list of tasks to do.
        sex (str): One of male, female, or both, to indicate sex split.
    """
    __slots__ = ["_location_id", "_recipe", "_sex"]

    def __init__(self, location_id, recipe, sex):
        assert isinstance(location_id, int)
        assert isinstance(recipe, str)
        allowed_sexes = {"male", "female", "both"}
        assert sex in allowed_sexes
        assert recipe not in allowed_sexes

        self._location_id = location_id
        self._recipe = recipe
        self._sex = sex

    @property
    def location_id(self):
        return self._location_id

    @property
    def recipe(self):
        return self._recipe

    @property
    def sex(self):
        return self._sex

    def __hash__(self):
        return hash((self._location_id, self._recipe, self._sex))

    def __repr__(self):
        return f"RecipeIdentifier({self.location_id}, {self.recipe} {self.sex})"


def recipe_graph_from_settings(locations, settings, args):
    """
    This defines the full set of recipes that are the model.
    These may be a subset of all locations,
    and we may execute a subset of these.

    Args:
        locations (nx.Graph): A graph of locations in a hierarchy.
        settings (Configuration): The EpiViz-AT Form (in form.py)
        args (Namespace|SimpleNamespace): Parsed arguments.

    Returns:
        nx.DiGraph: Each node is a RecipeIdentifier. Edges denote dependency
        on a previous transform. The graph has a key called "root" that
        tells you the first node.
    """
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
            drill = location_id_from_start_and_finish(locations, drill_start, drill_end)
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
        setup_task = [RecipeIdentifier(drill[0], "bundle_setup", "both")]

    tasks = setup_task + [
        RecipeIdentifier(drill_location, "estimate_location", "both")
        for drill_location in drill
    ]
    task_pairs = list(zip(tasks[:-1], tasks[1:]))
    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(tasks)
    task_graph.add_edges_from(task_pairs)
    # Add a custom graph attribute to record the tree root.
    task_graph.graph["root"] = tasks[0]
    return task_graph


def execution_ordered(graph):
    """For either a recipe graph or a task graph, this orders the nodes
    such that they go depth-first. This is chosen so that the data
    has the most locality during computation."""
    assert "root" in graph.graph, "Expect graph root to be in its dictionary."
    return nx.dfs_preorder_nodes(graph, graph.graph["root"])


def location_specific_settings(locations, settings, args, recipe_id):
    """
    This takes a modeler's description of how the model should be set up,
    as described in settings and command-line arguments, and translates
    it into what choices apply to this particular recipe. Modelers discuss
    plans in terms of what rules apply to which level of the Cascade,
    so this works in those terms, not in terms of individual tasks
    within a recipe.

    Adds ``settings.model.parent_location_id``,
    ``settings.model.grandparent_location_id``,
    and ``settings.model.children``.
    There is a grandparent location only if there is a grandparent recipe,
    so a drill starting halfway will not have a grandparent location.
    There are child locations for the last task though.

    Args:
        locations (nx.DiGraph): Location hierarchy
        settings: Settings from EpiViz-AT
        args (Namespace|SimpleNamespace): Command-line arguments
        recipe_id (RecipeIdentifier): Identifies what happens at
            this location.

    Returns:
        Settings for this job.
    """
    parent_location_id = recipe_id.location_id
    predecessors = list(locations.predecessors(parent_location_id))
    if predecessors:
        grandparent_location_id = predecessors[0]
    else:
        grandparent_location_id = None

    if settings.model.is_field_unset("drill_sex"):
        # An unset drill sex gets all data.
        sexes = [1, 2, 3]
    else:
        # Setting to male or female pulls in "both."
        sexes = [settings.model.drill_sex, 3]

    policies = policies_from_settings(settings)
    model_options = make_model_options(locations, parent_location_id, settings)
    if args.num_samples:
        sample_cnt = args.num_samples
    else:
        sample_cnt = policies["number_of_fixed_effect_samples"]

    local_settings = EstimationParameters(
        settings=settings,
        policies=policies,
        children=list(sorted(locations.successors(parent_location_id))),
        parent_location_id=parent_location_id,
        grandparent_location_id=grandparent_location_id,
        # This is a list of [1], [3], [1,3], [2,3], [1,2,3], not [1,2].
        sexes=sexes,
        number_of_fixed_effect_samples=sample_cnt,
        model_options=model_options,
    )
    local_settings.data_access = _ParameterHierarchy(**dict(
        gbd_round_id=policies["gbd_round_id"],
        decomp_step=policies["decomp_step"],
        modelable_entity_id=settings.model.modelable_entity_id,
        model_version_id=settings.model.model_version_id,
        settings_file=args.settings_file,
        bundle_file=args.bundle_file,
        bundle_id=settings.model.bundle_id,
        bundle_study_covariates_file=args.bundle_study_covariates_file,
        tier=2 if args.skip_cache else 3,
        age_group_set_id=policies["age_group_set_id"],
        with_hiv=policies["with_hiv"],
        cod_version=settings.csmr_cod_output_version_id,
        location_set_version_id=settings.location_set_version_id,
        add_csmr_cause=settings.model.add_csmr_cause,
    ))
    local_settings.run = _ParameterHierarchy(**dict(
        no_upload=args.no_upload,
        db_only=args.db_only,
        num_processes=args.num_processes,
        pdb=args.pdb,
    ))
    substeps = list()
    if settings.policies.fit_strategy == "fit_fixed_then_fit":
        substeps.append("initial_guess_from_fit_fixed")
    substeps.extend([
        "compute_initial_fit",
        "compute_draws_from_parent_fit",
        "save_predictions"
    ])
    return substeps, local_settings

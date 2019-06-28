"""
Specification for what parameters are used at what location within
the Cascade.
"""
from types import SimpleNamespace

import networkx as nx

from cascade.core import getLoggers
from cascade.core.parameters import ParameterProperty, _ParameterHierarchy
from cascade.input_data import InputDataError
from cascade.input_data.configuration.builder import policies_from_settings
from cascade.input_data.configuration.sex import SEX_ID_TO_NAME, SEX_NAME_TO_ID
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

    def __eq__(self, other):
        if not isinstance(other, RecipeIdentifier):
            return False
        return all(getattr(self, x) == getattr(other, x)
                   for x in ["location_id", "recipe", "sex"])

    def __hash__(self):
        return hash((self._location_id, self._recipe, self._sex))

    def __repr__(self):
        return f"RecipeIdentifier({self.location_id}, {self.recipe}, {self.sex})"


class JobIdentifier(RecipeIdentifier):
    __slots__ = ["_location_id", "_recipe", "_sex", "_name"]

    def __init__(self, recipe_identifier, name):
        self._name = name
        super().__init__(
            recipe_identifier.location_id,
            recipe_identifier.recipe,
            recipe_identifier.sex,
        )

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        return RecipeIdentifier.__eq__(self, other) and self.name == other.name

    def __hash__(self):
        return hash((self._location_id, self._recipe, self._sex, self.name))

    def __repr__(self):
        return f"RecipeIdentifier({self.location_id}, {self.recipe}, {self.sex}, {self.name})"


def recipe_graph_from_settings(locations, settings, args):
    """
    This defines the full set of recipes that are the model.
    These may be a subset of all locations,
    and we may execute a subset of these.

    Args:
        locations (nx.DiGraph): A graph of locations in a hierarchy.
        settings (Configuration): The EpiViz-AT Form (in form.py)
        args (Namespace|SimpleNamespace): Parsed arguments.

    Returns:
        nx.DiGraph: Each node is a RecipeIdentifier. Edges denote dependency
        on a previous transform. The graph has a key called "root" that
        tells you the first node.
    """
    if not settings.model.is_field_unset("drill") and settings.model.drill == "drill":
        recipe_graph = drill_recipe_graph(locations, settings, args)
    else:
        recipe_graph = global_recipe_graph(locations, settings, args)

    for recipe_identifier in recipe_graph.nodes:
        local_settings = location_specific_settings(locations, settings, args, recipe_identifier)
        recipe_graph.nodes[recipe_identifier]["local_settings"] = local_settings

    return recipe_graph


def drill_recipe_graph(locations, settings, args):
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
    MATHLOG.info(f"drill nodes {', '.join(str(d) for d in drill)}")
    drill = list(drill)
    drill_sex = SEX_ID_TO_NAME[settings.model.drill_sex]
    if args.skip_cache:
        setup_task = []
    else:
        setup_task = [RecipeIdentifier(drill[0], "bundle_setup", drill_sex)]
    recipes = setup_task + [
        RecipeIdentifier(drill_location, "estimate_location", drill_sex)
        for drill_location in drill
    ]
    recipe_pairs = list(zip(recipes[:-1], recipes[1:]))
    recipe_graph = nx.DiGraph(root=recipes[0])
    recipe_graph.add_nodes_from(recipes)
    recipe_graph.add_edges_from(recipe_pairs)
    # Add a custom graph attribute to record the tree root.
    recipe_graph.graph["root"] = recipes[0]
    return recipe_graph


def global_recipe_graph(locations, settings, args):
    """
    Constructs the graph of recipes.

    Args:
        locations (nx.DiGraph): Root node in the data, and each node
            has a level.
        settings: The global settings object.
        args (Namespace|SimpleNamespace): Command-line arguments.

    Returns:
        nx.DiGraph: Each node is a RecipeIdentifier.
    """
    assert "root" in locations.graph
    if settings.model.split_sex == "most_detailed":
        split_sex = max([locations.nodes[nl]["level"] for nl in locations.nodes])
    else:
        split_sex = int(settings.model.split_sex)
    global_node = RecipeIdentifier(locations.graph["root"], "estimate_location", "both")
    recipe_graph = nx.DiGraph(root=global_node)
    # Start with bundle setup
    if not args.skip_cache:
        bundle_setup = RecipeIdentifier(0, "bundle_setup", "both")
        recipe_graph.graph["root"] = bundle_setup
        recipe_graph.add_edge(bundle_setup, global_node)
    else:
        recipe_graph.graph["root"] = global_node

    global_recipe_graph_add_estimations(locations, recipe_graph, split_sex)

    return recipe_graph


def global_recipe_graph_add_estimations(locations, recipe_graph, split_sex):
    """There are estimations for every location and for both sexes below
    the level where we split sex. This modifies the recipe graph in place."""
    # Follow location hierarchy, splitting into male and female below a level.
    for start, finish in locations.edges:
        if "level" not in locations.nodes[finish]:
            raise RuntimeError(
                "Expect location graph nodes to have a level property")
        finish_level = locations.nodes[finish]["level"]
        if finish_level == split_sex:
            for finish_sex in ["male", "female"]:
                recipe_graph.add_edge(
                    RecipeIdentifier(start, "estimate_location", "both"),
                    RecipeIdentifier(finish, "estimate_location", finish_sex),
                )
        elif finish_level > split_sex:
            for same_sex in ["male", "female"]:
                recipe_graph.add_edge(
                    RecipeIdentifier(start, "estimate_location", same_sex),
                    RecipeIdentifier(finish, "estimate_location", same_sex),
                )
        else:
            recipe_graph.add_edge(
                RecipeIdentifier(start, "estimate_location", "both"),
                RecipeIdentifier(finish, "estimate_location", "both"),
            )


def execution_ordered(graph):
    """For either a recipe graph or a task graph, this orders the nodes
    such that they go depth-first. This is chosen so that the data
    has the most locality during computation."""
    assert "root" in graph.graph, "Expect to find G.graph['root']"
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
        sexes = list(SEX_ID_TO_NAME.keys())
    else:
        # Setting to male or female pulls in "both."
        sexes = [settings.model.drill_sex, SEX_NAME_TO_ID["both"]]

    policies = policies_from_settings(settings)
    model_options = make_model_options(locations, parent_location_id, settings)
    if args.num_samples:
        sample_cnt = args.num_samples
    else:
        sample_cnt = policies["number_of_fixed_effect_samples"]

    local_settings = EstimationParameters(
        settings=settings,
        policies=SimpleNamespace(**policies),
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
    return local_settings


class Job:
    def __init__(self, name, recipe_identifier, local_settings):
        self.name = name
        self.recipe = recipe_identifier
        self.local_settings = local_settings
        if name != "compute_draws_from_parent_fit":
            self.multiplicity = 1
        else:
            self.multiplicity = local_settings.number_of_fixed_effect_samples

    @property
    def job_identifier(self):
        return JobIdentifier(self.recipe, self.name)


def recipe_to_jobs(recipe_identifier, local_settings):
    """Given a recipe, return a list of jobs that must be done in order.

    Args:
        recipe_identifier (RecipeIdentifier): A data struct that specifies
            what a modeler thinks of as one estimation.
        local_settings (Namespace|SimpleNamespace): These are settings that
            have been localized to apply to a particular location.

    Returns:
        List[Job]: A list of jobs to run in order. Could make it a graph,
        but that's unnecessary.
    """
    sub_jobs = list()
    if recipe_identifier.recipe == "bundle_setup":
        bundle_setup = Job("bundle_setup", recipe_identifier, local_settings)
        sub_jobs.append(bundle_setup)
    elif recipe_identifier.recipe == "estimate_location":
        if local_settings.policies.fit_strategy == "fit_fixed_then_fit":
            sub_jobs.append(Job("fit_fixed_then_fit", recipe_identifier, local_settings))
        sub_jobs.extend([
            Job(job_name, recipe_identifier, local_settings)
            for job_name in [
                "compute_initial_fit",
                "compute_draws_from_parent_fit",
                "save_predictions"
            ]])
    else:
        raise RuntimeError(f"Unknown recipe identifier {recipe_identifier}")
    return sub_jobs


def recipe_graph_to_job_graph(recipe_graph):
    recipe_edges = dict()  # recipe_identifier -> (input node, output node)
    job_graph = nx.DiGraph()
    for copy_identifier in recipe_graph.nodes:
        job_list = recipe_graph.node[copy_identifier]["job_list"]
        if len(job_list) < 1:
            raise RuntimeError(f"Recipe {copy_identifier} doesn't have any sub-jobs.")
        job_ids = [job_node.job_identifier for job_node in job_list]
        job_graph.add_nodes_from(
            (jid, dict(job=add_job))
            for (jid, add_job) in zip(job_ids, job_list)
        )
        job_graph.add_edges_from(zip(job_ids[:-1], job_ids[1:]))
        recipe_edges[copy_identifier] = dict(input=job_ids[0], output=job_ids[-1])

        if copy_identifier == recipe_graph.graph["root"]:
            job_graph.graph["root"] = job_ids[0]

    assert "root" in job_graph.graph, "Could not find a root node for the graph"
    job_graph.add_edges_from([
        (recipe_edges[start]["output"], recipe_edges[finish]["input"])
        for (start, finish) in recipe_graph.edges
    ])
    return job_graph


def job_graph_from_settings(locations, settings, args):
    recipe_graph = recipe_graph_from_settings(locations, settings, args)
    for node in recipe_graph:
        jobs = recipe_to_jobs(node, recipe_graph.nodes[node]["local_settings"])
        recipe_graph.nodes[node]["job_list"] = jobs
    return recipe_graph_to_job_graph(recipe_graph)

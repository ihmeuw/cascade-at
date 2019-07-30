"""
This is a fairly sophisticated generator of EpiViz-AT settings.
It makes settings that are internally-consistent and possibly-stochastic.
You decide how much to specify:

 1. You can choose parameters you would like to fix to particular values.
 2. Pass those parameters to the settings generator.
 3. The settings generator creates a random settings file starting from
    the selections you chose.

If you don't choose an initial set, then all the parameters are random.
"""
from configparser import ConfigParser
from copy import deepcopy
from pprint import pprint
from textwrap import dedent

import networkx as nx
from gridengineapp import execution_ordered
from numpy.random import RandomState

from cascade.executor.cascade_plan import recipe_graph_from_settings
from cascade.executor.dismodel_main import DismodAT
from cascade.input_data.configuration import SettingsError
from cascade.input_data.db.configuration import json_settings_to_frozen_settings

BASE_CASE = {
    "model": {
        "random_seed": 899736881,
        "default_age_grid": "0 .1 .2 1 5 10 20 30 40 50 60 70 80 90 100",
        "default_time_grid": "2000 2005 2010",
        "birth_prev": 0,
        "ode_step_size": 1,
        "minimum_meas_cv": 0.1,
        "rate_case": "iota_pos_rho_pos",
        "data_density": "gaussian",
        "constrain_omega": 1,
        "fix_cov": 1,
        "split_sex": 1,
        "modelable_entity_id": 23514,
        "bundle_id": 4325,
        "model_version_id": 267890,
        "add_calc_emr": 1,
        "drill": "drill",
        "drill_sex": 2,
        "drill_location_start": 1,
        "drill_location_end": 40,
        "zero_sum_random": [
            "iota",
            "rho",
            "chi"
        ],
        "quasi_fixed": 0,
        "addl_ode_stpes": "0.015625 0.03125 0.0625 0.125 0.25 0.5",
        "title": "LBW / running on NAME",
        "drill_location": 40,
        "bound_random": 0.1459
    },
    "max_num_iter": {
        "fixed": 100,
        "random": 100
    },
    "print_level": {
        "fixed": 5,
        "random": 0
    },
    "accept_after_max_steps": {
        "fixed": 5,
        "random": 5
    },
    "students_dof": {
        "priors": 5,
        "data": 5
    },
    "log_students_dof": {
        "priors": 5,
        "data": 5
    },
    "eta": {
        "priors": 0.01,
        "data": 0.01
    },
    "config_version": 1,
    "gbd_round_id": 5,
    "csmr_cod_output_version_id": 90,
    "csmr_mortality_output_version_id": 90,
    "location_set_version_id": 429,
    "min_cv": [
        {
            "cascade_level_id": 1,
            "value": 0.1
        }
    ],
    "tolerance": {
        "fixed": 1e-10
    },
}


def grid(non_negative, rng, name, single_age=False):
    grid_case = {
        "default": {
            "value": {
                "density": "gaussian",
                "mean": 0.1,
                "std": 0.1
            },
            "dage": {
                "density": "gaussian",
                "mean": 0,
                "std": 0.1
            },
            "dtime": {
                "density": "gaussian",
                "mean": 0,
                "std": 0.1
            }
        },
        "age_time_specific": rng.choice([0, 1], p=[0.25, 0.75], name=f"{name}.at_specific"),
    }
    if non_negative:
        # The min has to be > 0?
        grid_case["default"]["value"]["min"] = rng.choice([1e-4], name=f"{name}.min")
        grid_case["default"]["value"]["mean"] = grid_case["default"]["value"]["min"] + 1e-2
    age_choice = rng.choice([0, 1, 2], name=f"{name}.age_cnt")
    if age_choice == 1:
        grid_case["age_grid"] = "0"
    elif age_choice == 2:
        grid_case["age_grid"] = "0 1 5 10"
    # else 0 use default ages
    if single_age:
        grid_case["age_grid"] = "0"
    time_choice = rng.choice([0, 1, 2], name=f"{name}.time_cnt")
    if time_choice == 1:
        grid_case["time_grid"] = "2000"
    elif time_choice == 2:
        grid_case["time_grid"] = "2000 2005 2010"
    # else 0 use default times
    return grid_case


def rate_grid(rate, rng):
    single_age = rate == "pini"
    grid_case = grid(True, rng, single_age=single_age, name=f"{rate}")
    grid_case["rate"] = rate
    return grid_case


def add_random_effects(children, rate_name, rng):
    random_effects = list()
    single_age = rate_name == "pini"
    if children:
        for child in children:
            grid_case = grid(False, rng, single_age=single_age, name=f"re.{rate_name}.{child}")
            grid_case["rate"] = rate_name
            grid_case["location"] = child
            random_effects.append(grid_case)
    else:
        grid_case = grid(False, rng, single_age=single_age, name=f"re.{rate_name}")
        grid_case["rate"] = rate_name
        grid_case["location"] = None
        random_effects.append(grid_case)
    return random_effects


def covariate(study_country, covariate_idx, nonzero_rates, rng, name):
    assert study_country in ["study", "country"]
    grid_case = grid(False, rng, name=f"{name}")
    grid_case["mulcov_type"] = rng.choice(["meas_std", "meas_value", "rate_value"], name=f"{name}.covtype")
    if grid_case["mulcov_type"] == "rate_value":
        chosen = rng.choice(nonzero_rates, name=f"{name}.rate")
        grid_case["measure_id"] = dict(iota=41, rho=7, pini=5, chi=9, omega=16).get(chosen)
    else:
        grid_case["measure_id"] = 15
    grid_case[f"{study_country}_covariate_id"] = covariate_idx
    grid_case["transformation"] = 0
    return grid_case


def add_covariates(case, study_id, country_id, nonzero_rates, rng):
    for ckind, covariates in [("study", study_id), ("country", country_id)]:
        scovariates = list()
        for make_cov in covariates:
            include = rng.choice([False, True], p=[0.7, 0.3], name=f"{ckind}.{make_cov}")
            if include:
                scovariates.append(
                    covariate(ckind, make_cov, nonzero_rates, rng, name=f"{ckind}.{make_cov}"))
        if scovariates:
            case[f"{ckind}_covariate"] = scovariates


def create_settings(choices, locations=None):
    """
    Makes a random settings, as though EpiViz-AT made it.

    Args:
        choices (SettingsChoice): Create with ``RandomState(2324234)``.
        locations (nx.DiGraph): List of locations.

    Returns:
        dict: A dictionary that looks like parsed JSON from EpiViz-AT.
    """
    if isinstance(choices, RandomState):
        rng = SettingsChoices(choices)
    elif isinstance(choices, SettingsChoices):
        rng = choices
    else:
        raise ValueError(f"choices object should be a SettingsChoices or rng not {type(choices)}")
    locations = locations if locations else make_locations(4)

    has = dict()
    for rate, likely in [("iota", 0.1), ("rho", 0.7), ("omega", 0), ("chi", 0.1), ("pini", 0.7)]:
        has[rate] = rng.choice([False, True], p=[likely, 1 - likely], name=rate)
    nonzero_rates = [x for (x, y) in has.items() if y]

    case = deepcopy(BASE_CASE)
    if has["pini"]:
        case["model"]["birth_prev"] = 1

    case["model"]["add_calc_emr"] = rng.choice([0, 1], name="emr")
    case["model"]["constrain_omega"] = rng.choice([0, 1], p=[0.1, 0.9], name="constrain_omega")

    # Guarantee at least one nonzero rate, after constraint of omega.
    if case["model"]["constrain_omega"] == 1:
        has["omega"] = True
        if not any(has[x] for x in ["iota", "rho", "chi"]):
            has["iota"] = True
    else:
        if sum(has.values()) == 0:
            has["omega"] = True

    case["rate"] = list()
    for rate in nonzero_rates:
        case["rate"].append(rate_grid(rate, rng))

    location_root = 1
    # Use the last location as the drill end possibility.
    last_location = list(nx.topological_sort(locations))[-1]
    drill = sorted(nx.ancestors(locations, last_location))
    start_loc = rng.choice(list(range(len(drill))), name="drill_start")
    end_loc = rng.choice(list(range(len(drill[start_loc:]))), name="drill_end")
    case["model"]["drill_location_start"] = drill[start_loc]
    case["model"]["drill_location_end"] = drill[start_loc:][end_loc]
    print(f"drill {drill} with start {drill[start_loc]} end {drill[start_loc:][end_loc]}")

    rate_specifies_re_by_location = which_random_effects(nonzero_rates, rate, rng)

    add_chosen_random_effects(case, location_root, locations, rate_specifies_re_by_location, rng)

    study_covariates = [0, 11, 1604]
    country_covariates = [156, 1998]
    add_covariates(case, study_covariates, country_covariates, nonzero_rates, rng)
    try:
        config = json_settings_to_frozen_settings(case, 267890)
    except SettingsError:
        pprint(case, indent=2)
        raise
    return config


def add_chosen_random_effects(case, location_root, locations, rate_specifies_re_by_location, rng):
    random_effects = list()
    # Iterates over every set of parent and children in the graph.
    for rate_name, by_location in rate_specifies_re_by_location:
        if by_location:
            # You have to do _all_ of them.
            for parent, children in nx.bfs_successors(locations, location_root):
                random_effects.extend(add_random_effects(children, rate_name, rng))
        else:
            add_random_effects(None, rate_name, rng)
    # Only add to dict if there are some?
    if random_effects:
        case["random_effect"] = random_effects


def which_random_effects(nonzero_rates, rate, rng):
    # We specify priors of random effects either
    # a) Once for the whole hierarchy, or
    # b) For every single location in the hierarchy.
    # Choose this by rate
    rate_specifies_re_by_location = dict()
    for rate_name in nonzero_rates:
        choice = rng.choice(["none", "all", "every"], name=f"re.{rate_name}")
        # none means no random effects.
        # all means every single location.
        # every means each location gets a new one.
        if rate in {"all", "every"}:
            rate_specifies_re_by_location[rate_name] = choice == "every"
        else:
            pass  # Don't record this rate as needing a random effect.
    return rate_specifies_re_by_location


class SettingsChoices:
    """
    This class will fix certain choices to given values. Its input is
    a list of key-value pairs, set up as for a section of a ConfigParser,
    although this adds the section title for you.
    """
    def __init__(self, rng=None, settings=None):
        if isinstance(settings, str):
            parser = ConfigParser()
            parser.read_string("[settings]\n" + dedent(settings))
            self.answers = parser["settings"]
        else:
            self.answers = dict()
        self.rng = rng
        self.random = list()

    def choice(self, choices, name=None, p=None):
        if name in self.answers:
            value = self.answers[name]
            # ConfigParser returns strings.
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == "true":
                        return True
                    elif value.lower() == "false":
                        return False
            return value
        else:
            assert self.rng, f"Unexpected setting chosen {name} {choices}"
            self.random.append((name, choices))
            assert len(choices) > 0
            return self.rng.choice(choices, p=p)


def make_locations(depth):
    """Creates locations of given depth as a balanced tree. Root is 1."""
    arity = 3
    zero_based = nx.balanced_tree(arity, depth, nx.DiGraph())
    locations = nx.relabel_nodes(zero_based, {i: i + 1 for i in range(len(zero_based))})
    for lidx, n in enumerate(locations.nodes):
        locations.nodes[n]["location_name"] = str(lidx)
    return locations


def create_local_settings(rng=None, settings=None, locations=None):
    """Make a local settings object, all the way from the EpiViz-AT form."""
    rng = rng if rng else RandomState(3242352)
    if isinstance(rng, RandomState):
        choices = SettingsChoices(rng, settings)
    else:
        choices = rng
    # skip-cache says to use tier 2, not tier 3 so that we don't need CSMR there.
    args = DismodAT.add_arguments().parse_args(["--skip-cache"])
    depth = 4
    locations = locations if locations else make_locations(depth)
    settings = create_settings(choices, locations)
    print(f"location count {len(locations)}")
    recipe_graph = recipe_graph_from_settings(locations, settings, args)
    print(f"recipe graph nodes={len(recipe_graph)}")
    # skip-cache also turns off the first, non-estimation, job.
    jobs = list(execution_ordered(recipe_graph))
    assert len(jobs) > 0
    job_choice = choices.choice(list(range(len(jobs))), name="job_idx")
    local_settings = recipe_graph.nodes[jobs[job_choice]]["local_settings"]
    return local_settings, locations
